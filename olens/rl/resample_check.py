"""How much of the lens's novelty FVE (what its bullets add over the realized continuation) do Qwen's own resampled
continuations already cover? If most of it, the lens is a "plausible next text" generator and the realized-continuation
reward can't push it anywhere new.

For each val position from train.py build and a few of its stage-A reads (train.py eval):
    m0 = FVE(C ∪ B) - FVE(C)            C: realized continuation, all 32 prefixes (the RL reward's free atoms)
    m1 = FVE(C ∪ R ∪ B) - FVE(C ∪ R)    R: k continuations Qwen samples at that position, prefixes 1,2,4,8,16,32 each
    m2 = same with R' = another val position's resamples at the same layer (control: free atoms that don't fit)

    PYTHONPATH=olens/src:olens/rl python olens/rl/resample_check.py RUN_DIR READS.jsonl sample [--k 8]
    PYTHONPATH=olens/src:olens/rl python olens/rl/resample_check.py RUN_DIR READS.jsonl fit [--reads-per-pos 4]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from olens.ar import AR
from olens.continuation import nnls
from olens.fve import Whitener
from olens.lens import BASE, load_base
from olens.readout import bullets
from reward import K_MAX, fve_batch
from safetensors.torch import load_file
from train import ROLLOUTS

PREFIXES = [1, 2, 4, 8, 16, 32]


def fve(z, atoms, check=False):
    """z [B, d], atoms: list of B tensors [k_i, d] -> FVE per row (zero-padded to one batch; zero rows are dropped). With
    check, also solves the first 64 rows with the exact per-row scipy NNLS and prints the largest gap."""
    k = max(len(a) for a in atoms)
    D = torch.zeros(len(atoms), k, z.shape[1], dtype=torch.float64)
    for i, a in enumerate(atoms):
        D[i, : len(a)] = a
    out = fve_batch(z.cuda(), D.cuda())
    if check:
        exact = nnls(z[:64].double(), D[:64])[2].numpy()
        print(f"Gram-Cholesky vs exact NNLS on {k}-atom rows: max |diff| {np.abs(exact - out[:64]).max():.2e}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("reads")
    ap.add_argument("stage", choices=["sample", "fit"])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--reads-per-pos", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    stage = lambda name: print(f"[{time.time() - t0:.0f}s] {name}", flush=True)
    run = Path(args.run)
    d = torch.load(run / "positions.pt", weights_only=False)
    reads = {r["idx"]: r for r in map(json.loads, Path(args.reads).read_text().splitlines())}
    val = [i for i in np.flatnonzero(d["split"] == "val") if i in reads]
    for i in val[:20]:  # positions.pt was rebuilt on this pod: same seed, so the same positions
        assert reads[i]["continuation"] == d["cont_text"][i], "positions differ from the ones the reads came from"

    stage("loading rollouts")
    t = load_file(hf_hub_download("agu18dec/local-workspace", ROLLOUTS, repo_type="dataset"))
    off, ids = t["offsets"].numpy(), t["ids"].numpy()
    prefixes = [ids[off[d["conv"][i]] : off[d["conv"][i]] + d["pos"][i] + 1].tolist() for i in val]

    stage("vllm")
    sample_file = run / "resample_check_samples.json"
    if args.stage == "sample":  # vLLM in its own process: next to it, moving HF Qwen onto the GPU stalled for 20+ min
        from vllm import LLM, SamplingParams, TokensPrompt

        llm = LLM(
            model=BASE, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85, enable_prefix_caching=True, max_num_seqs=256
        )
        # T=1, untruncated: how Agam's rollouts were sampled. 32 tokens = the free-atom window.
        params = SamplingParams(n=args.k, temperature=1.0, top_p=1.0, max_tokens=32, seed=0)
        outs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prefixes], params)
        sample_file.write_text(json.dumps([[list(c.token_ids) for c in o.outputs] for o in outs]))
        return
    samples = json.loads(sample_file.read_text())

    stage("sampled; loading HF")
    tok, model = load_base()
    ar = AR(model)
    W = {}
    R = []  # per position: [k * 6, d] whitened resample atoms
    for i, ss in zip(val, samples):
        L = int(d["layer"][i])
        if L not in W:
            W[L] = Whitener(L, device=model.device, dtype=torch.float32)
        P = ar.prefixes([s for s in ss if s], L)  # [k, <=32, d], zero past each sample's end
        cols = [p - 1 for p in PREFIXES if p <= P.shape[1]]
        A = P[:, cols].reshape(-1, P.shape[-1])
        A = A[A.abs().sum(1) > 0]
        R.append(W[L](A).cpu())
    stage("resample atoms done")
    Z = d["Z"][val].double()
    C = [d["C"][i].float() for i in val]
    by_layer = {}
    for n, i in enumerate(val):
        by_layer.setdefault(int(d["layer"][i]), []).append(n)
    rng = np.random.default_rng(0)
    ctrl = [rng.choice([m for m in by_layer[int(d["layer"][i])] if m != n]) for n, i in enumerate(val)]

    stage("fitting")
    f_C = fve(Z, C)
    f_R = fve(Z, R)
    f_CR = fve(Z, [torch.cat([c, r]) for c, r in zip(C, R)], check=True)
    f_CRc = fve(Z, [torch.cat([c, R[m]]) for c, m in zip(C, ctrl)])
    stage("base fits done")
    rows = []
    for j in range(args.reads_per_pos):
        texts = [reads[i]["reads"][j]["text"] for i in val]
        bls = [bullets(x)[:K_MAX] for x in texts]
        B = []
        for i, bl in zip(val, bls):
            L = int(d["layer"][i])
            spans = [tok(" " + b, add_special_tokens=False).input_ids[:32] for b in bl] or [[tok.eos_token_id]]
            B.append(W[L](ar(spans, L)).cpu() * (1.0 if bl else 0.0))
        f_B = fve(Z, B)
        f_CB = fve(Z, [torch.cat([c, b]) for c, b in zip(C, B)])
        f_CRB = fve(Z, [torch.cat([c, r, b]) for c, r, b in zip(C, R, B)])
        f_CRcB = fve(Z, [torch.cat([c, R[m], b]) for c, m, b in zip(C, ctrl, B)])
        for n, i in enumerate(val):
            rows.append({
                "idx": int(i), "layer": int(d["layer"][i]), "read": j, "bullets": bls[n],
                "f_C": f_C[n], "f_R": f_R[n], "f_CR": f_CR[n], "f_CRc": f_CRc[n], "f_B": f_B[n],
                "m0": f_CB[n] - f_C[n], "m1": f_CRB[n] - f_CR[n], "m2": f_CRcB[n] - f_CRc[n],
            })  # fmt: skip
    out = run / "resample_check"
    out.mkdir(exist_ok=True)
    with (out / "rows.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({k: float(v) if isinstance(v, np.floating) else v for k, v in r.items()}, ensure_ascii=False) + "\n")
    with (out / "samples.jsonl").open("w") as f:
        for i, ss in zip(val, samples):
            f.write(json.dumps({"idx": int(i), "context": d["ctx_text"][i], "realized": d["cont_text"][i][:200],
                                "resamples": [tok.decode(s) for s in ss]}, ensure_ascii=False) + "\n")  # fmt: skip
    keys = ["f_C", "f_R", "f_CR", "f_CRc", "f_B", "m0", "m1", "m2"]
    summary = {"all": {k: float(np.mean([r[k] for r in rows])) for k in keys}}
    for L in sorted(by_layer):
        summary[f"L{L}"] = {k: float(np.mean([r[k] for r in rows if r["layer"] == L])) for k in keys}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["all"]), flush=True)
    m = summary["all"]
    print(f"survives resamples m1/m0 = {m['m1'] / m['m0']:.3f}, control m2/m0 = {m['m2'] / m['m0']:.3f}")


if __name__ == "__main__":
    main()
