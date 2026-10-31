"""NLA reads of the author_blog_drop_inline prompts with ceselder/qwen3.6-27b-nla-rl (Kylie, 2026-10-31), at the O-lens's
read positions. That NLA reads layer 42 only, and its AV is a full model (`av_base`: Qwen3.6-27B with the warm-start AV
LoRA merged) plus an RL LoRA (`av_rl_adapters/iter_000800`, the last step, 78% held-out FVE), so it can't run on our
one-engine reader (olens.nla, base Qwen + one LoRA). Instead, three plain transformers stages on one GPU pod:

  capture  base Qwen3.6-27B, residual at hidden_states[42] at every read position (scripts/hs_at_positions.py)
  decode   av_base + RL adapter: the `actor` prompt (chat template, thinking off), the activation added Karvonen-style at
           the marker (h <- h + |h| v/|v|, output of decoder block 1, prefill only), greedy, <= 200 tokens, stop at
           </explanation> (olens.nla's settings)
  score    the AR (`ar_reconstructor`: 43-block backbone, no final norm, value head) on "Summary of the following text:
           <text>{explanation}</text> <summary>": the predicted activation -> FVE, the NLA paper's form (p and v at norm
           sqrt(d); mu and E||v - mu||^2 from this run's activations). No pooled ruler: the agu18dec whiteners have no L42.

    uv run --with transformers --with jinja2 python -m scripts.author_probe.nla_l42 inputs RUN   # local: RUN/hs_inputs.jsonl
    # pod (scripts/sky/nla_l42.yaml): capture, then decode + score -> RUN/nla_raw.jsonl
    uv run python -m scripts.author_probe.nla_l42 finalize RUN   # local: RUN/reads.jsonl (batch reads format) + transcripts
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

REPO = "ceselder/qwen3.6-27b-nla-rl"
ADAPTER = "av_rl_adapters/iter_000800"
LAYER = 42
READS = [Path("data/runs/olens_reads/2026-10-30_author-blog"), Path("data/runs/olens_reads/2026-10-30_author-blog-b")]
INPUTS = Path("data/runs/cfbench_evals/2026-10-30_author-blog-drop-inline-inputs-b2")
D = 5120


def load(p: Path) -> list[dict]:
    return [json.loads(line) for line in Path(p).read_text().splitlines()]


def requests() -> dict[str, dict]:
    """The O-lens read requests of the eval's pairs (first reads run wins), with prompt_ids and positions."""
    need = {c["pair"] for c in load(INPUTS / "cases.jsonl")}
    out = {}
    for d in READS:
        for r in load(d / "requests.jsonl"):
            if r["id"] in need:
                out.setdefault(r["id"], r)
    return out


def inputs(run: Path):
    reqs = requests()
    rows = [{"id": pair, "ids": r["prompt_ids"], "sites": {"h": r["positions"]}, "cand": []} for pair, r in sorted(reqs.items())]
    (run / "hs_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(reqs)} prompts, {sum(len(r['positions']) for r in reqs.values())} positions -> {run / 'hs_inputs.jsonl'}")


def decode_and_score(run: Path, repo_dir: str, batch: int = 96):
    """Pod: hs/h.npy (from capture) -> nla_raw.jsonl rows {id, position, text, fve, cos}."""
    import torch
    import yaml
    from peft import PeftModel
    from safetensors.torch import load_file
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load(run / "hs_inputs.jsonl")
    acts = np.load(run / "hs" / "h.npy")[:, 0]  # [N, d], the one saved layer
    keys = [(r["id"], p) for r in rows for p in r["sites"]["h"]]
    assert len(keys) == len(acts), (len(keys), acts.shape)
    meta = yaml.safe_load(Path(repo_dir, "nla_meta.yaml").read_text())
    tok = AutoTokenizer.from_pretrained(f"{repo_dir}/av_base")
    prompt = meta["prompt_templates"]["actor"].format(injection_char=meta["tokens"]["injection_char"])
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
    ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    slot = ids.index(int(meta["tokens"]["injection_token_id"]))
    assert (ids[slot - 1], ids[slot + 1]) == (meta["tokens"]["injection_left_neighbor_id"], meta["tokens"]["injection_right_neighbor_id"])

    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(f"{repo_dir}/av_base", dtype=torch.bfloat16, device_map="cuda")
    av = PeftModel.from_pretrained(base, f"{repo_dir}/{ADAPTER}").eval()
    print(f"AV loaded in {time.time() - t0:.0f}s", flush=True)
    layers = base.model.layers if hasattr(base.model, "layers") else base.model.language_model.layers
    cur: dict = {}

    def hook(_m, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] > slot:  # prefill only
            v = cur["v"].to(h.device, h.dtype)
            hs = h[:, slot]
            h[:, slot] = hs + hs.norm(dim=-1, keepdim=True) * v / v.norm(dim=-1, keepdim=True)
        return out

    layers[1].register_forward_hook(hook)
    texts = []
    t0 = time.time()
    for i in range(0, len(acts), batch):
        cur["v"] = torch.from_numpy(acts[i : i + batch].astype(np.float32))
        x = torch.tensor([ids] * len(cur["v"]), device="cuda")
        with torch.no_grad():
            g = av.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=200, do_sample=False,
                            stop_strings=["</explanation>"], tokenizer=tok, pad_token_id=tok.pad_token_id or tok.eos_token_id)  # fmt: skip
        for row in g[:, len(ids) :]:
            t = tok.decode(row, skip_special_tokens=True)
            t = t.split("<explanation>", 1)[-1].split("</explanation>", 1)[0].strip()
            texts.append(t)
        print(f"decoded {len(texts)}/{len(acts)} ({len(texts) / (time.time() - t0):.1f}/s)", flush=True)
    del av, base
    torch.cuda.empty_cache()

    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(f"{repo_dir}/ar_reconstructor", dtype=torch.bfloat16).to("cuda").eval()
    m.model.norm = nn.Identity()  # no final norm in the checkpoint: the head reads the last block's output
    head = nn.Linear(D, D, bias=False).to("cuda", torch.float32)
    head.load_state_dict(load_file(f"{repo_dir}/ar_reconstructor/value_head.safetensors"))
    print(f"AR loaded in {time.time() - t0:.0f}s", flush=True)
    tmpl = meta["prompt_templates"]["critic"]
    tok.padding_side = "left"
    preds = []
    for i in range(0, len(texts), 64):
        batch_texts = [tmpl.format(explanation=t) for t in texts[i : i + 64]]
        enc = tok(batch_texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            h = m.model(**enc).last_hidden_state[:, -1].float()
            preds.append(head(h).cpu())
    p = torch.cat(preds)
    v = torch.from_numpy(acts.astype(np.float32))
    s = math.sqrt(D)
    pn, vn = p * (s / p.norm(dim=-1, keepdim=True)), v * (s / v.norm(dim=-1, keepdim=True))
    mu = vn.mean(0)
    fve = 1 - ((pn - vn) ** 2).sum(-1) / ((vn - mu) ** 2).sum(-1).mean()
    cos = (pn * vn).sum(-1) / D
    with open(run / "nla_raw.jsonl", "w") as f:
        for (pair, pos), t, a, c in zip(keys, texts, fve.tolist(), cos.tolist(), strict=True):
            f.write(json.dumps({"id": pair, "token_index": pos, "text": t, "fve": round(a, 4), "cos": round(c, 4)}) + "\n")
    print(f"scored {len(texts)}: mean FVE {fve.mean():.3f}, median {fve.median():.3f}", flush=True)


def finalize(run: Path):
    """nla_raw.jsonl -> reads.jsonl in the eval inputs' reads format (msg_index, field, char_offset, token, layer, text,
    fve), with the O-lens reads' offsets for the same (id, token_index)."""
    off = {}
    for r in load(INPUTS / "reads/reads.jsonl"):
        off[r["id"], r["token_index"]] = {k: r[k] for k in ("msg_index", "role", "field", "char_offset", "token", "prefix") if k in r}
    out = [{"id": r["id"], "token_index": r["token_index"], **off[r["id"], r["token_index"]], "layer": LAYER, "text": r["text"],
            "fve": r["fve"], "cos": r["cos"]} for r in load(run / "nla_raw.jsonl")]  # fmt: skip
    (run / "reads.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    reqs = requests()
    (run / "requests.jsonl").write_text("".join(json.dumps(reqs[k]) + "\n" for k in sorted(reqs)))
    print(f"{len(out)} reads -> {run / 'reads.jsonl'}")


if __name__ == "__main__":
    cmd, run = sys.argv[1], Path(sys.argv[2])
    {"inputs": inputs, "finalize": finalize, "decode": lambda r: decode_and_score(r, sys.argv[3])}[cmd](run)
