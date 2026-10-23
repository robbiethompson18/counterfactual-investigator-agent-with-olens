"""Extra free atoms for the v2 reward (train.py --free v2): credit the lens only for what neither the transcript nor plausible
next text explains. Per position, on top of the realized continuation's 32 prefixes (positions.pt "C"):

    context: AR of the last 1, 2, 4, 8, 16, 32 tokens up to and including the position, and of the 32 before those
    resamples: k continuations of 32 tokens Qwen samples at the position (T=1), AR of prefixes 1, 2, 4, 8, 16, 32 of each

resample_check.py found the resamples alone cover ~40% of the lens's novelty over the realized continuation; the reads it
kept were mostly restated context and text more than 32 tokens ahead.

Two processes, because HF Qwen next to vLLM in one process stalled (resample_check.py):
    PYTHONPATH=olens/src:olens/rl python olens/rl/free_atoms.py RUN sample [--k 8]   # vLLM -> RUN/resamples.json
    PYTHONPATH=olens/src:olens/rl python olens/rl/free_atoms.py RUN atoms            # HF + AR -> RUN/free_v2.pt {"X": [N, 55, d]}
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from olens.ar import AR
from olens.fve import Whitener
from olens.lens import BASE, load_base
from safetensors.torch import load_file
from train import LAYERS, ROLLOUTS

PREFIXES = [1, 2, 4, 8, 16, 32]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("stage", choices=["sample", "atoms"])
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    run = Path(args.run)
    d = torch.load(run / "positions.pt", weights_only=False)
    N = len(d["layer"])
    t0 = time.time()
    if args.stage == "sample":
        from vllm import LLM, SamplingParams, TokensPrompt

        t = load_file(hf_hub_download("agu18dec/local-workspace", ROLLOUTS, repo_type="dataset"))
        off, ids = t["offsets"].numpy(), t["ids"].numpy()
        prefixes = [ids[off[c] : off[c] + p + 1].tolist() for c, p in zip(d["conv"], d["pos"])]
        llm = LLM(BASE, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85, enable_prefix_caching=True, max_num_seqs=256)
        params = SamplingParams(n=args.k, temperature=1.0, top_p=1.0, max_tokens=32, seed=0)  # how the rollouts were sampled
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prefixes], params)
        (run / "resamples.json").write_text(json.dumps([[list(c.token_ids) for c in o.outputs] for o in outs]))
        print(f"sampled {N} x {args.k} in {time.time() - t0:.0f}s", flush=True)
        return

    samples = json.loads((run / "resamples.json").read_text())
    tok, model = load_base()
    ar = AR(model)
    n_ctx = len(PREFIXES) + 1
    X = torch.zeros(N, n_ctx + args.k * len(PREFIXES), d["H"].shape[1], dtype=torch.float16)
    for L in LAYERS:
        W = Whitener(L, device=model.device, dtype=torch.float32)
        js = np.flatnonzero(d["layer"] == L)
        # context: suffixes ending at the position, plus the 32-token window before the longest one
        spans = [d["ctx_ids"][i][-n:] for i in js for n in PREFIXES] + [d["ctx_ids"][i][-64:-32] for i in js]
        live = torch.tensor([len(x) > 0 for x in spans], device=model.device)[:, None]  # no window before a short context
        spans = [x or [tok.eos_token_id] for x in spans]
        A = W(torch.cat([ar(spans[s : s + 256], L) for s in range(0, len(spans), 256)])) * live
        ctx = torch.cat([A[: len(js) * len(PREFIXES)].view(len(js), len(PREFIXES), -1), A[len(js) * len(PREFIXES) :, None]], 1)
        X[js, :n_ctx] = ctx.half().cpu()
        for s in range(0, len(js), 8):  # 8 positions x k samples per AR forward
            b = js[s : s + 8]
            seqs = [samples[i][j] or [tok.eos_token_id] for i in b for j in range(args.k)]
            P = ar.prefixes(seqs, L)  # [8k, <=32, d], zero past each sample's end
            cols = [p - 1 for p in PREFIXES if p <= P.shape[1]]
            R = W(P[:, cols]) * (P[:, cols].abs().sum(-1, keepdim=True) > 0)  # keep padding rows at 0 after whitening
            R = torch.nn.functional.pad(R, (0, 0, 0, len(PREFIXES) - len(cols)))
            X[b, n_ctx:] = R.view(len(b), args.k * len(PREFIXES), -1).half().cpu()
        print(f"L{L}: {len(js)} positions, {time.time() - t0:.0f}s", flush=True)
    torch.save({"X": X, "k": args.k, "prefixes": PREFIXES, "n_ctx": n_ctx}, run / "free_v2.pt")


if __name__ == "__main__":
    main()
