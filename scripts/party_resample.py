"""Resample the third-party-politics organism from the positions its O-lens reads were taken at, to
check whether a party the lens names but the rollout never mentions is a real low-probability branch
or a lens hallucination. Setup and results: docs/party-resample.md.

    python scripts/party_resample.py prep                    # local: sample-0 rows -> SOURCE
    python scripts/party_resample.py gen COND MODEL_DIR      # pod, vLLM env: COND in organism/base
    PYTHONPATH=olens/src:transcript-viewer/pod:scripts python scripts/party_resample.py lens ADAPTER_DIR
                                                             # pod, olens env: sampled lens reads

`gen`: for each read position t inside the reply, N_PREFIX continuations from prefix = the rollout's
own `ids[:t+1]` (no re-tokenization), plus N_FRESH rollouts from the prompt. Positions in the
end-of-user-turn zone (before the reply starts) are forced chat-template tokens, so the fresh
rollouts are the resample for them. Sampling matches scripts/organism_rollouts.py: T=1, top_p 0.95,
top_k 20, thinking off, 700 new tokens counted from the start of the reply.

`lens`: N_LENS reads at T=1 per (position, layer, condition) through the viewer's read path
(transcript-viewer/pod/server.py), on the organism's activations (hook-LoRA) and on base Qwen's.
Not yet run to completion: the first attempt wrote nothing in 60+ minutes and was killed; it now writes
per chunk.
"""

import json
import sys
import time
from pathlib import Path

ORGANISM = "auditbench:third_party_politics_td_r16"
READS = Path("data/runs/organisms/olens_reads.jsonl")
OUT = Path("data/runs/party_resample")
SOURCE = OUT / "source.jsonl"
N_PREFIX, N_FRESH, N_LENS = 64, 200, 8
MAX_NEW = 700
LAYERS = (24, 32, 40, 48, 56)


def load_source():
    return [json.loads(line) for line in SOURCE.open()]


def prompt_len(row):
    """The reply starts after the empty think block's closing "</think>", "\\n\\n"."""
    return row["tokens"].index("</think>") + 2


def prep():
    OUT.mkdir(parents=True, exist_ok=True)
    keep = ("organism", "prompt_id", "system", "user", "sample", "completion", "ids", "positions", "tokens")
    rows = [json.loads(line) for line in READS.open()]
    rows = [r for r in rows if r["organism"] == ORGANISM and r["sample"] == 0 and r["kind"] == "trigger"]
    with SOURCE.open("w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in keep}) + "\n")
    print(f"{len(rows)} rows -> {SOURCE}")


def gen(cond, model_dir):
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=model_dir,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
        seed=0,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    tok = llm.get_tokenizer()
    reqs, meta = [], []
    for row in load_source():
        ids, P = row["ids"], prompt_len(row)
        assert tok.decode(ids[P - 4 : P]) == "<think>\n\n</think>\n\n", tok.decode(ids[P - 4 : P])
        starts = [("fresh", P - 1, N_FRESH)] + [("prefix", t, N_PREFIX) for t in row["positions"] if P <= t < len(ids) - 1]
        for source, t, n in starts:
            reqs.append((TokensPrompt(prompt_token_ids=ids[: t + 1]), MAX_NEW - (t + 1 - P), n))
            meta.append({"cond": cond, "prompt_id": row["prompt_id"], "source": source, "t": t, "prefix": tok.decode(ids[P : t + 1])})
    t0 = time.time()
    params = [SamplingParams(n=n, temperature=1.0, top_p=0.95, top_k=20, max_tokens=mx) for _, mx, n in reqs]
    outs = llm.generate([p for p, _, _ in reqs], params)
    path = OUT / f"continuations_{cond}.jsonl"
    n_tok = 0
    with path.open("w") as f:
        for m, o in zip(meta, outs, strict=True):
            for k, c in enumerate(o.outputs):
                n_tok += len(c.token_ids)
                f.write(json.dumps({**m, "sample": k, "text": c.text, "n_tokens": len(c.token_ids), "finish": c.finish_reason}) + "\n")
    print(f"{cond}: {sum(len(o.outputs) for o in outs)} continuations, {n_tok} tokens, {time.time() - t0:.0f}s -> {path}", flush=True)


def lens(adapter_dir):
    import server  # transcript-viewer/pod/server.py
    import torch
    from olens.ar import AR
    from olens.fve import WHITENERS
    from olens.lens import AguLens, Sampling, capture, load_base
    from olens.lora import HookLoRA
    from olens.readout import Readout, bullets

    tok, model = load_base()
    agu = AguLens(tok, model)
    server.S.update(tok=tok, model=model, agu=agu, readout=Readout(agu, AR(model)))
    server.BATCH = 192
    if not WHITENERS.exists():  # same fallback as scripts/organism_olens.py
        print("no whiteners: reads are not FVE-scored", flush=True)
        server.score = lambda texts, vecs, layers: [[{"text": t, "bullets": bullets(t)} for t in ts] for ts in texts]
    smp = Sampling(temperature=1.0, top_p=0.95, top_k=64, max_new_tokens=160, max_bullets=4)
    lora = HookLoRA(model, adapter_dir, model.device)
    vecs, layers, where = [], [], []
    for row in load_source():
        for cond in ("organism", "base"):
            lora.active = cond == "organism"
            acts = capture(model, row["ids"], LAYERS)
            lora.active = False
            for p in row["positions"]:
                for L in LAYERS:
                    vecs.append(acts[L][p].cpu())
                    layers.append(L)
                    where.append((row["prompt_id"], cond, p, L))
            del acts
    lora.remove()
    # in chunks, written as they finish: the one-shot version ran >60 min on an H200 with no output (2026-10-21)
    t0, chunk = time.time(), server.BATCH // N_LENS
    with (OUT / "lens_samples.jsonl").open("w") as f:
        for i in range(0, len(vecs), chunk):
            res = server.read_many(torch.stack(vecs[i : i + chunk]), layers[i : i + chunk], smp, n=N_LENS, seed=i)
            for (pid, cond, p, L), samples in zip(where[i : i + chunk], res, strict=True):
                for k, rd in enumerate(samples):
                    f.write(json.dumps({"prompt_id": pid, "cond": cond, "pos": p, "layer": L, "sample": k, **rd}) + "\n")
            f.flush()
            print(f"lens: {i + len(res)}/{len(vecs)} vectors x {N_LENS} samples, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        prep()
    elif cmd == "gen":
        gen(sys.argv[2], sys.argv[3])
    elif cmd == "lens":
        lens(sys.argv[2])
