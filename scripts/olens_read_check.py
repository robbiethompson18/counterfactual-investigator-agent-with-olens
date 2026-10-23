"""Check scripts/olens_read_requests.py's vLLM decode + score against the HF reference (olens.Readout on AguLens):
same activations, greedy, 4 bullets; prints joint FVE per path and first-bullet agreement. Run on a pod after
runpod/olens_reads.sh setup:

    PYTHONPATH=olens/src:scripts python scripts/olens_read_check.py /workspace/merged/agu_space_rl600
"""

import statistics
import sys

import torch
from olens.ar import AR
from olens.lens import AguLens, load_base
from olens.readout import Readout
from olens.vllm_lens import VllmLens
from olens_read_requests import SAMPLING, capture, score
from transformers import AutoTokenizer

PROMPTS = [
    "The athlete Muhammad Ali plays the sport of",
    "My coworker keeps taking credit for my work in meetings. What should I do?",
    "I'm trapped in a burning warehouse and my phone is at 3%. Help.",
    "Can you call my mom and tell her I'll be late?",
]
LAYERS = [24, 44, 56]


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    vl = VllmLens(tok, sys.argv[1], gpu_memory_utilization=0.45, max_num_seqs=64)
    _, model = load_base()
    vl.model = model
    ar = AR(model)
    agu = AguLens(tok, model)
    ref = Readout(agu, ar)
    items = []
    for p in PROMPTS:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
        ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
        pos = list(range(len(ids) - 6, len(ids)))  # end-of-user-turn zone
        acts = capture(model, ids, LAYERS, pos)
        items += [(L, acts[L][k]) for k in range(len(pos)) for L in LAYERS]
    vecs = torch.stack([v for _, v in items])
    lays = [L for L, _ in items]
    ours = score(tok, ar, ref, [t[0] for t in vl.read_items(vecs, lays, SAMPLING)], vecs, lays)
    hf = [None] * len(items)  # HF generate, one batch per layer
    for L in LAYERS:
        idx = [i for i, x in enumerate(lays) if x == L]
        for i, t in zip(idx, agu.read(vecs[idx], L, sampling=SAMPLING)):
            hf[i] = ref.score(t[0], vecs[i], L)
    for L in LAYERS:
        a = [o[1] for o, x in zip(ours, lays) if x == L]
        b = [h["fve"] for h, x in zip(hf, lays) if x == L]
        print(f"L{L}: joint FVE vLLM+score {statistics.mean(a):.3f} | HF Readout {statistics.mean(b):.3f} (n={len(a)})")
    same = sum(bool(o[0]) and bool(h["bullets"]) and o[0][0]["text"] in h["bullets"] for o, h in zip(ours, hf))
    print(f"top-FVE vLLM bullet among HF's bullets: {same}/{len(items)}")
    print("example:", [b["text"] for b in ours[0][0]], "|", hf[0]["bullets"])


if __name__ == "__main__":  # vLLM spawns its engine process, which re-imports this module
    main()
