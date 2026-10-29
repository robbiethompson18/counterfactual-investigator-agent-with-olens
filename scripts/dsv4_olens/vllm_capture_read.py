"""One-engine DSV4 Flash O-lens reads: read.py's capture + first-reply logits on vLLM, then vllm_read.py's lens stage on the
same engine. Replaces the HF transformers capture pass (second venv, states.pt handoff) of the two-stage pipeline.

Stage 1, adapter off: per request, one `max_tokens=1` prefill of prompt + reply with vllm-metamodels' capture hooks
at the read layers (`extra_args={"output_residual_stream": [L], "capture_positions": [...]}`; the hc_capture plugin in
hc_inject/ folds V4's 4-stream layer output into the [4 * 4096] rows read.py's Capture hook saw) with
`prompt_logprobs` for read.py's "reject" (log P of that sequence's reply after </think>: <rejected/>, or the actual
reply with --actual-reply) and the top-10 at the first reply token, plus one prefill of prompt +
<request_fully_accepted/> for "accept". Positions, rows and the logits.jsonl / reads_L*.jsonl formats are read.py's
and vllm_read.py's: the rows come from read.render / read.read_positions, the lens stage is vllm_read.decode_states.

    python vllm_capture_read.py --model /workspace/dsv4/w/v4flash --lens /workspace/dsv4/w/v4lens --requests requests.jsonl \\
        --out out/ --layers 30 --dense --actual-reply --read-arms preview,public [--limit 20] [--check-against REF_DIR]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("VLLM_LENS_MERGE_ALLOW_TP", "1")

sys.path.insert(0, str(Path(__file__).parent))

import torch
from read import ACCEPT, ASSISTANT, NO_THINK, REJECT, USER, read_positions, render
from vllm_read import build_engine, check_against, decode_states


def acts(out, n_layers, positions):
    """vllm-metamodels' residual_stream for one request -> [n_layers, P, 16384] in the requested position order."""
    act = torch.as_tensor(out.activations["residual_stream"])
    got = [int(p) for p in out.activations["positions"]]
    if act.dim() == 2:
        act = act[None]
    assert act.shape[0] == n_layers and act.shape[1] == len(got), (tuple(act.shape), n_layers, len(got))
    if got != list(positions):
        order = {p: i for i, p in enumerate(got)}
        act = act[:, [order[p] for p in positions]]
    return act.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--lora", default=None, help="converted adapter dir (default LENS/lora_vllm)")
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="30", help="layers to capture and read (carriers exist for 15, 18, ..., 42)")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--chunk", type=int, default=2048, help="reads per lens generate() call")
    ap.add_argument("--batch", type=int, default=32, help="transcripts per capture generate() call")
    ap.add_argument(
        "--max-model-len", type=int, default=1024, help="longest prompt + tag; prefilled in one chunk so the hooks see every position"
    )
    ap.add_argument("--dense", action="store_true", help="read.py's --dense positions")
    ap.add_argument(
        "--shared-prompt",
        action="store_true",
        help="every prompt token (system, user turn, assistant header) except the request's `query` text in the user turn; no reply",
    )
    ap.add_argument("--actual-reply", action="store_true", help="read the request's own reply, not a canonical <rejected/>")
    ap.add_argument("--read-arms", default="public", help="arms whose transcripts get lens reads (all get logits)")
    ap.add_argument("--public-first", action="store_true")
    ap.add_argument("--save-states", action="store_true", help="also write states.pt in read.py's format")
    ap.add_argument("--skip-lens", action="store_true", help="capture + logits only")
    ap.add_argument("--check-against", default=None, help="dir with another run's reads_L*.jsonl + logits.jsonl: print parity")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    layers = sorted(int(x) for x in a.layers.split(","))
    reqs = [json.loads(line) for line in Path(a.requests).read_text().splitlines()][: a.limit or None]
    read_arms = set(a.read_arms.split(","))
    tok = AutoTokenizer.from_pretrained(a.model)
    timings = {}
    t = time.time()
    llm = build_engine(a.model, a.tp, a.eager, a.max_model_len, max_num_batched_tokens=max(8192, a.max_model_len), chunked_prefill=False)
    timings["engine_s"] = time.time() - t

    # 1. capture + logits, adapter off. Per request two prefills, as read.py's batch of two: [prompt + reply] (the capture, with
    # prompt_logprobs: read.py's "reject" is log P of this sequence's reply, i.e. <rejected/> or, with --actual-reply, the actual
    # reply; its top-10 at position n is top_first) and [prompt + accept tag] with prompt_logprobs for "accept".
    states, logit_rows, pos_rows = [], [], []
    seen_system: set[str] = set()
    t = time.time()
    for b0 in range(0, len(reqs), a.batch):
        prompts, params, parsed = [], [], []
        for r in reqs[b0 : b0 + a.batch]:
            prompt, reply = render(r["messages"])
            full = prompt + (reply if a.actual_reply else REJECT)
            ids, offs, pos = read_positions(tok, full, None, a.dense, seen_system if a.dense else None)
            if a.shared_prompt:  # the part of the prompt every item shares (scripts/af_v8x_olens_requests.py)
                q0 = full.index(r["query"], full.rindex(USER))
                q1 = q0 + len(r["query"])
                pos = [i for i, (s, e) in enumerate(offs) if 0 < i and s < len(prompt) and not (s < q1 and e > q0)]
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            assert ids[: len(p_ids)] == p_ids, "prompt tokenization is not a prefix of prompt+reply"
            assert len(ids) + 8 <= a.max_model_len, f"{r['id']}: {len(ids)} tokens > --max-model-len {a.max_model_len}"
            acc = p_ids + tok(ACCEPT, add_special_tokens=False)["input_ids"]
            capture = r["arm"] in read_arms and bool(pos)
            extra = {"output_residual_stream": layers, "capture_positions": pos} if capture else None
            prompts += [{"prompt_token_ids": ids}, {"prompt_token_ids": acc}]
            params += [
                SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=10, extra_args=extra),
                SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=0),
            ]
            parsed.append((r, prompt, reply, full, offs, pos, len(p_ids), ids, acc, capture))
        outs = iter(llm.generate(prompts, params, use_tqdm=False))
        for r, prompt, reply, full, offs, pos, n, ids, acc, capture in parsed:
            lp = {}
            for name, seq in (("reject", ids), ("accept", acc)):
                o = next(outs)
                pl = o.prompt_logprobs  # pl[i] predicts token i from tokens < i (pl[0] is None); the actual token is always included
                lp[name] = float(sum(pl[j][seq[j]].logprob for j in range(n, len(seq))))
                if name == "reject":  # top-10 at the first reply token (prompt_logprobs=10 on this request only)
                    top = sorted(pl[n].items(), key=lambda kv: -kv[1].logprob)[:10]
                    lp["top_first"] = [(tok.decode([tid]), round(float(x.logprob), 3)) for tid, x in top]
                    if capture:
                        act = acts(o, len(layers), pos)  # [n_layers, P, 16384] bf16
                        for j, p in enumerate(pos):
                            s, e = offs[p]
                            row = {"id": r["id"], "arm": r["arm"], "item": r["item"], "pos": p, "char": s}
                            who = (
                                "assistant"
                                if s >= len(prompt) - len(ASSISTANT + NO_THINK)
                                else "user"
                                if s >= full.rindex(USER)
                                else "system"
                            )
                            row |= {"role": who, "token": full[s:e]}
                            pos_rows.append(row | {"ctx_left": full[max(0, s - 80) : e], "ctx_right": full[e : e + 40]})
                            states.append(act[:, j])
            logit_rows.append({"id": r["id"], "arm": r["arm"], "item": r["item"], "reply": reply[:200], "positive": r["positive"], **lp})
        last, done = logit_rows[-1], min(b0 + a.batch, len(reqs))
        print(f"{done}/{len(reqs)} {last['id']} rej {last['reject']:.2f} acc {last['accept']:.2f} {time.time() - t:.0f}s", flush=True)
    timings["capture_logits_s"] = time.time() - t
    print(f"capture+logits {len(reqs)} requests, {len(pos_rows)} positions, {timings['capture_logits_s']:.0f}s", flush=True)
    with open(out / "logits.jsonl", "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in logit_rows)
    S = torch.stack(states) if states else torch.empty(0, len(layers), 4 * 4096, dtype=torch.bfloat16)  # [N, n_layers, 16384]
    if a.save_states:
        torch.save({"layers": layers, "states": S.view(len(S), len(layers), 4, -1).to(torch.bfloat16), "rows": pos_rows}, out / "states.pt")
    (out / "timings.json").write_text(json.dumps(timings, indent=1))
    if a.skip_lens or not pos_rows:
        return

    # 2. lens generations, adapter on: vllm_read.py's stage over the in-memory states
    lora = LoRARequest("v4lens", 1, a.lora or str(Path(a.lens) / "lora_vllm"))
    order = None if not a.public_first else [lambda r: r["arm"] != "public", lambda r: r["id"].split(":")[0] in ("v2", "v3")]
    reads_path = out / f"reads_L{','.join(map(str, layers))}.jsonl"
    t = time.time()
    decode_states(
        llm, lora, a.lens, pos_rows, lambda L: S[:, layers.index(L)].float(), layers, reads_path, a.samples, a.max_new, a.chunk, order
    )
    timings["lens_s"] = time.time() - t
    timings["reads"] = len(pos_rows) * len(layers)
    (out / "timings.json").write_text(json.dumps(timings, indent=1))
    print(f"timings {timings}", flush=True)
    if a.check_against:
        ref = Path(a.check_against)
        check_against(reads_path, ref / reads_path.name, label="ref")


if __name__ == "__main__":
    main()
