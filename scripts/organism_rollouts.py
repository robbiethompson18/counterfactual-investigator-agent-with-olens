"""Sample rollouts from existing Qwen3.6-27B LoRA model organisms, and from the base model on the same
prompts, for browsing and for O-lens reads later.

One bf16 base model; each organism is applied as a hook-LoRA (olens/src/olens/lora.py) while its
prompts are sampled, then removed. vLLM's LoRA runtime is inert on this base (Agam's
servers/README), so this uses HF generate. Thinking off, as the organisms were trained.

Prompts: scripts/organism_prompts.json. Per family: `trigger` prompts (should show the behaviour),
optional `control` prompts (minimal edits that shouldn't), and the shared `neutral` prompts. The base
model gets every family's prompts under the same system prompt.

    PYTHONPATH=olens/src python scripts/organism_rollouts.py [n_samples] [family,...]
writes data/runs/organisms/rollouts.jsonl (appends; skips (organism, system, user) prompts already done).
"""

import json
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from olens.lens import load_base
from olens.lora import HookLoRA

PROMPTS = json.loads(Path("scripts/organism_prompts.json").read_text())
OUT = Path("data/runs/organisms/rollouts.jsonl")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ONLY = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
MAX_NEW, BATCH = 700, 48

# organism key -> (repo, subfolder or None)
REPOS = {
    "auditbench": "agu18dec/auditbench-model-orgs",
    "farzanah": "farzanah/qwen3.6-27b-sandbagging-sft-{}",
    "taboo": "EvilScript/Qwen3_6-27B-taboo-{}",
    "latentmo": "latentMO/reward-hacking-opsd",
}


def adapter_dir(key):
    src, name = key.split(":")
    if src in ("auditbench", "latentmo"):
        root = snapshot_download(REPOS[src], allow_patterns=[f"{name}/*"])
        return Path(root) / name
    return Path(snapshot_download(REPOS[src].format(name)))


def system_for(spec, kind):
    """`system` is "prism", "none", or {kind: text} when trigger and control differ only in the system
    prompt (the eval-sandbagging family); neutral prompts then get the control one."""
    s = spec["system"]
    if isinstance(s, dict):
        return s["control" if kind == "neutral" else kind]
    return PROMPTS["prism_system"] if s == "prism" else None


def rows_for(family, spec, organism):
    for kind in ("trigger", "control"):
        for i, user in enumerate(spec.get(kind, [])):
            yield {
                "organism": organism,
                "family": family,
                "kind": kind,
                "prompt_id": f"{family}/{kind}{i}",
                "system": system_for(spec, kind),
                "user": user,
            }
    for i, user in enumerate(PROMPTS["neutral"]):
        yield {
            "organism": organism,
            "family": family,
            "kind": "neutral",
            "prompt_id": f"neutral{i}",
            "system": system_for(spec, "neutral"),
            "user": user,
        }


@torch.no_grad()
def sample(tok, model, rows, seed):
    texts = []
    for r in rows:
        msgs = ([{"role": "system", "content": r["system"]}] if r["system"] else []) + [{"role": "user", "content": r["user"]}]
        texts += [tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)] * N
    out = []
    torch.manual_seed(seed)
    for b in range(0, len(texts), BATCH):
        enc = tok(texts[b : b + BATCH], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(
            **enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=1.0, top_p=0.95, top_k=20, pad_token_id=tok.pad_token_id
        )
        new = gen[:, enc.input_ids.shape[1] :]
        for row in new:
            ids = [t for t in row.tolist() if t != tok.pad_token_id]
            out.append((tok.decode(ids, skip_special_tokens=True), len(ids), tok.eos_token_id not in ids and len(ids) >= MAX_NEW))
    return [
        {**r, "sample": s, "completion": c, "n_tokens": n, "truncated": t}
        for (r, s), (c, n, t) in zip([(r, s) for r in rows for s in range(N)], out, strict=True)
    ]


def main():
    tok, model = load_base()
    tok.padding_side = "left"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["organism"], r["system"], r["user"]) for r in map(json.loads, OUT.open())} if OUT.exists() else set()
    families = {f: s for f, s in PROMPTS["families"].items() if ONLY is None or f in ONLY}
    jobs = [(org, list(rows_for(f, s, org))) for f, s in families.items() for org in s["organisms"]]
    # the base model gets every prompt once; neutral prompts recur per family, so dedupe on (system, user)
    base = {(r["system"], r["user"]): r for f, s in families.items() for r in rows_for(f, s, "base")}
    jobs.append(("base", list(base.values())))
    for seed, (org, rows) in enumerate(jobs):
        rows = [r for r in rows if (org, r["system"], r["user"]) not in done]
        if not rows:
            continue
        t0 = time.time()
        lora = HookLoRA(model, adapter_dir(org), model.device) if org != "base" else None
        if lora:
            lora.active = True
        res = sample(tok, model, rows, seed)
        if lora:
            lora.remove()
        with OUT.open("a") as f:
            f.writelines(json.dumps(r) + "\n" for r in res)
        print(f"{org}: {len(res)} rollouts, {sum(r['n_tokens'] for r in res)} tokens, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
