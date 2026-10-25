"""Persistent GLM-5.3 O-lens worker for the 4xB200 box.

Boots Celeste's engine once (olens_glm53_usage.GlmOlensEngine: nvidia/GLM-5.3-NVFP4, W4A16 Marlin experts, fp8 KV, vllm-metamodels glm53),
then processes job files dropped into JOBS/ in name order. A job is a .pt dict
  {"ids": [[int]], "pos": [[int]], "salt": str}  (token ids of each rendered view, positions to read)
and produces JOBS/<name>.out.pt  {"h51_norm": [[float]], "rl": [[[greedy, s1..s4]]]}.
Per job: capture block-51 of the unmerged base at every position, merge the RL lens, read greedy + 4 samples at T=1, unmerge.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import time
import traceback
import uuid

import olens_glm53_usage as U
import torch


@torch.no_grad()
def capture_h51(eng, token_ids: list[list[int]], positions: list[list[int]]) -> list[torch.Tensor]:
    """GlmOlensEngine.capture_h51 with a unique vLLM cache_salt per request: with prefix caching on, no capture request can hit
    another request's cached blocks (a cached token is never recomputed, so its block-51 activation would not be captured)."""
    from vllm import SamplingParams

    if eng.merged is not None:
        eng.set_lens(None)
    reqs, sps = [], []
    for ids, pos in zip(token_ids, positions):
        reqs.append({"prompt_token_ids": [int(t) for t in ids], "cache_salt": uuid.uuid4().hex})
        sps.append(
            SamplingParams(
                max_tokens=1, temperature=0.0, extra_args={"output_residual_stream": [U.READ_LAYER], "capture_positions": list(pos)}
            )
        )
    outs = eng.llm.generate(reqs, sps, use_tqdm=False)
    res = []
    for o, pos in zip(outs, positions):
        a = o.activations["residual_stream"]
        assert a.shape[0] == 1 and a.shape[1] == len(pos) and a.shape[-1] == U.HIDDEN, (tuple(a.shape), len(pos))
        res.append(a[0].float().cpu())
    return res


@torch.no_grad()
def read_rl(eng, vecs: torch.Tensor, seed: int, cap: int = 16, max_tokens: int | None = None) -> list[list[str]]:
    """greedy + 4 samples at T=1 per activation (Celeste's protocol), submitted as one generate() so the batch stays full.
    Same requests as GlmOlensEngine.verbalize; the overwrite check counts one replaced row per prefill (n=4 without prefix caching = 4)."""
    from vllm import SamplingParams

    gram = U.GrammarSpec(cap=cap).to_dict()
    reqs, sps = [], []
    for i in range(vecs.shape[0]):
        sv = [U.steer_embed(vecs[i], eng.mpos)]
        # Celeste's lens_cache_salt=lens-sha256: the greedy bullet and the 4 samples of ONE position share the steered prefill
        tag = "lens-" + hashlib.sha256(sv[0].activations.numpy().tobytes()).hexdigest()
        for n, temp in ((1, 0.0), (4, 1.0)):
            reqs.append({"prompt_token_ids": eng.prompt_ids})
            sps.append(
                SamplingParams(
                    n=n,
                    temperature=temp,
                    top_p=1.0,
                    max_tokens=max_tokens or cap + 2,
                    seed=seed * 1_000_003 + 2 * i + (n > 1),
                    skip_special_tokens=True,
                    extra_args={"apply_steering_vectors": sv, "bullet_grammar": gram, "lens_cache_salt": tag},
                )
            )
    eng._stats()
    outs = eng.llm.generate(reqs, sps, use_tqdm=False)
    replaced = [int(x.get("rows_replaced", 0)) for x in eng._stats()]
    n_req = vecs.shape[0]
    assert all(n_req <= r <= 5 * n_req for r in replaced), f"embedding overwrite rows {replaced} vs {n_req} activations"
    return [[outs[2 * i].outputs[0].text.strip()] + [c.text.strip() for c in outs[2 * i + 1].outputs] for i in range(n_req)]


def job_bench(eng, a, p, job):
    """re-read a done job's first N positions with the current engine: timing only (output -> <name>.out.pt with the reads)."""
    src = torch.load(job["src"])
    acts = capture_h51(eng, src["ids"][:8], src["pos"][:8])
    flat = torch.cat(acts)[: job.get("n", 4096)]
    eng.set_lens(a.rl_lens)
    t0 = time.time()
    reads = read_rl(eng, flat, seed=1, max_tokens=job.get("max_tokens"))
    dt = time.time() - t0
    eng.set_lens(None)
    torch.save({"reads": reads, "dt": dt}, p.replace(".job.pt", ".out.pt"))
    U.log(f"[worker] BENCH {os.path.basename(p)}: {flat.shape[0]} pos read {dt:.1f}s ({flat.shape[0] / dt:.1f} pos/s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--rl-lens", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--max-model-len", type=int, default=33024)
    ap.add_argument("--gpu-mem", type=float, default=0.88)
    ap.add_argument("--read-chunk", type=int, default=8192, help="positions per verbalize() call")
    ap.add_argument("--cap-chunk", type=int, default=64, help="transcripts per capture_h51() call")
    ap.add_argument("--max-num-seqs", type=int, default=512)
    ap.add_argument(
        "--prefix-cache", action="store_true", help="vLLM prefix caching with the fork's KV salting (as Celeste's later dense units)"
    )
    ap.add_argument(
        "--grammar", choices=["batched", "adapter"], default="batched", help="batched = grammar_lp.BatchedBulletGrammarLP (same masks)"
    )
    a = ap.parse_args()
    if a.prefix_cache:
        import vllm

        class PrefixCachedLLM(vllm.LLM):  # GlmOlensEngine hardcodes enable_prefix_caching=False
            def __init__(self, *args, **kw):
                kw["enable_prefix_caching"] = True
                super().__init__(*args, **kw)

        vllm.LLM = PrefixCachedLLM
    if a.grammar == "batched":
        from grammar_lp import BatchedBulletGrammarLP

        U.GlmBulletGrammarLP = BatchedBulletGrammarLP  # GlmOlensEngine passes this module global to LLM(logits_processors=[...])
    eng = U.GlmOlensEngine(
        a.model_dir, tp=4, gpu_mem=a.gpu_mem, max_model_len=a.max_model_len, max_num_seqs=a.max_num_seqs, max_num_batched_tokens=32768
    )
    U.log("[worker] ready")
    with open(os.path.join(a.jobs, "READY"), "w") as f:
        f.write(str(time.time()))
    while True:
        todo = sorted(p for p in glob.glob(os.path.join(a.jobs, "*.job.pt")) if not os.path.exists(p.replace(".job.pt", ".out.pt")))
        if not todo:
            if os.path.exists(os.path.join(a.jobs, "STOP")):
                U.log("[worker] STOP")
                return
            time.sleep(5)
            continue
        p = todo[0]
        try:
            run_job(eng, a, p)
        except Exception:  # noqa: BLE001
            U.log(f"[worker] FAILED {p}\n{traceback.format_exc()}")
            os.replace(p, p + ".failed")
            try:
                eng.set_lens(None)
            except Exception as e:  # noqa: BLE001
                U.log(f"[worker] could not restore the base weights: {e!r}")


def run_job(eng, a, p):
    if True:
        job = torch.load(p)
        if job.get("bench"):
            job_bench(eng, a, p, job)
            return
        t0 = time.time()
        ids, pos = job["ids"], job["pos"]
        acts = []
        for c in range(0, len(ids), a.cap_chunk):
            acts += capture_h51(eng, ids[c : c + a.cap_chunk], pos[c : c + a.cap_chunk])
        t1 = time.time()
        flat = torch.cat(acts)
        eng.set_lens(a.rl_lens)
        reads = []
        for c in range(0, flat.shape[0], a.read_chunk):
            v = flat[c : c + a.read_chunk]
            reads += read_rl(eng, v, seed=job.get("seed", 0) + c)
        eng.set_lens(None)
        t2 = time.time()
        out, k = {"h51_norm": [], "rl": []}, 0
        for x in acts:
            out["h51_norm"].append(x.norm(dim=-1).tolist())
            out["rl"].append(reads[k : k + x.shape[0]])
            k += x.shape[0]
        tmp = p.replace(".job.pt", ".out.tmp")
        torch.save(out, tmp)
        os.replace(tmp, p.replace(".job.pt", ".out.pt"))
        U.log(
            f"[worker] {os.path.basename(p)}: {len(ids)} views {flat.shape[0]} pos capture {t1 - t0:.0f}s read {t2 - t1:.0f}s "
            f"({flat.shape[0] / max(t2 - t0, 1e-9):.1f} pos/s)"
        )


if __name__ == "__main__":
    main()
