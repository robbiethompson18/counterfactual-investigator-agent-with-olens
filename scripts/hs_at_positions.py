"""HF transformers forward passes over pre-tokenized prompts: residual stream at given token positions and the next-token
log-probs of given candidate tokens at the last position. Needs only torch, transformers and numpy, so it runs in any GPU
venv (e.g. the one-engine reader's), with the prompts tokenized elsewhere.

    python scripts/hs_at_positions.py INPUTS.jsonl OUT_DIR [--layer-stride 2 | --layers 32,40,52] [--chunk N]

INPUTS.jsonl: one row per prompt, {id, ids (token ids), sites: {name: [token index, ...]}, cand (token ids)}. OUT_DIR gets
<site>.npy ([all rows' positions in order, saved layers, d] fp16, hidden_states index 0, stride, ..., n_layers, or
--layers) and logp.jsonl ({id, logp: log-softmax over the full vocab at the last position, at each cand token}), plus
layers.json. With --chunk N the rows are done N at a time into OUT_DIR/part-K/ (same files), skipping parts already
there, so a killed job resumes; concatenate the parts in order.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM


def run(model, rows, out: Path, layers: list[int]):
    out.mkdir(parents=True, exist_ok=True)
    acts = {s: [] for s in rows[0]["sites"]}
    with open(out / "logp.jsonl.tmp", "w") as f:
        for i, r in enumerate(rows):
            with torch.no_grad():
                o = model(torch.tensor([r["ids"]], device=model.device), output_hidden_states=True)
            for s, pos in r["sites"].items():
                acts[s].append(np.stack([o.hidden_states[L][0, pos].float().cpu().numpy() for L in layers], 1).astype(np.float16))
            logp = torch.log_softmax(o.logits[0, -1].float(), -1)[r["cand"]].tolist()
            f.write(json.dumps({"id": r["id"], "logp": logp}) + "\n")
            if i % 50 == 0:
                print(out.name, i, len(rows), flush=True)
    for s, xs in acts.items():
        np.save(out / f"{s}.npy", np.concatenate(xs))
    (out / "layers.json").write_text(json.dumps(layers))
    (out / "logp.jsonl.tmp").rename(out / "logp.jsonl")  # last: a part is done once logp.jsonl exists
    print(f"saved {len(rows)} prompts, sites {list(acts)}, {len(layers)} layers to {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs")
    ap.add_argument("out")
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--layer-stride", type=int, default=2)
    ap.add_argument("--layers", help="comma-separated hidden_states indices, instead of --layer-stride")
    ap.add_argument("--chunk", type=int, help="rows per part (OUT/part-K/), resumable")
    a = ap.parse_args()
    rows = [json.loads(line) for line in Path(a.inputs).read_text().splitlines()]
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto").eval()
    n_layers = model.config.get_text_config().num_hidden_layers
    layers = list(range(0, n_layers + 1, a.layer_stride)) + ([n_layers] if n_layers % a.layer_stride else [])
    if a.layers:
        layers = [int(x) for x in a.layers.split(",")]
    if not a.chunk:
        return run(model, rows, Path(a.out), layers)
    for k in range(0, len(rows), a.chunk):
        part = Path(a.out) / f"part-{k // a.chunk:04d}"
        if not (part / "logp.jsonl").exists():
            run(model, rows[k : k + a.chunk], part, layers)


if __name__ == "__main__":
    main()
