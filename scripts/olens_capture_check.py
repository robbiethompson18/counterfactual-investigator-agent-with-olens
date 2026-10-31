"""Check that olens_read_requests.py reads the positions you meant: capture the residual at the first rows' read
positions with the one engine (no lens, no AR) at one block and compare it to activations saved elsewhere (e.g. by HF
transformers). Prints the cosine and norm ratio per position; cos ~ 1 means the rendering, positions and layer index agree.

    PYTHONPATH=olens/src:src python scripts/olens_capture_check.py REQUESTS.jsonl REF.npy --block 51

REF.npy: [n positions, d] for the first rows' positions in order. --block is the capture hook index (output of decoder
block L, HF's hidden_states[L + 1]).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from olens.one_engine import OneEngine
from transformers import AutoTokenizer

from counterfactual_investigator_agent_with_olens.reads import prepare  # untested on a pod since the move (2026-10-31)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("requests")
    ap.add_argument("ref")
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--base", default="Qwen/Qwen3.6-27B")
    ap.add_argument("--lens-lora", default="/workspace/lora_vllm/olens_space_rl600")
    args = ap.parse_args()
    ref = np.load(args.ref)
    tok = AutoTokenizer.from_pretrained(args.base)
    eng = OneEngine(tok, args.base, args.lens_lora, ar_lora=None, ar_root=None, max_num_seqs=8, max_model_len=8192)
    im_end, im_start = tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|im_start|>")
    got, rows = [], [json.loads(line) for line in Path(args.requests).read_text().splitlines()]
    for row in rows:
        _, ids, _, toks, positions, _, _ = prepare(tok, row, [args.block], im_end, im_start)
        got += [(toks[p], v) for p, v in zip(positions, eng.capture(ids, [args.block], positions)[args.block].numpy())]
        if len(got) >= len(ref):
            break
    for (t, v), r in zip(got, ref):
        nv, nr = np.linalg.norm(v), np.linalg.norm(r)
        print(f"{t!r:8} cos {np.dot(v, r) / nv / nr:.4f}  |capture|/|ref| {nv / nr:.3f}")


if __name__ == "__main__":
    main()
