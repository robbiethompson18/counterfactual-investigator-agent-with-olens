"""O-lens reads of activations with the model's own continuation projected out ("residual reads").

At a position p with activation h (layer L), fit the whitened h on the AR directions of the next n tokens Qwen actually
wrote (olens.continuation: every prefix 1..n, NNLS), then read the lens on

    h' = m·μ + γ·Σ^1/2 (z − β·fit),   z = Σ^-1/2 (h − μ)

β = 0, γ = 1, m = 1 is the ordinary read. β = 1 removes the fitted continuation, β = 2 pushes away from it; γ and m
move the injected direction away from the mean (the lens normalises its input, so a plain scale would do nothing).
Everything but the ordinary read is off-distribution for the lens. The `wrong` arm is the control: it removes the fit to
a length-matched continuation from another transcript, rescaled to the same norm as the true fit, so it moves h
as far but along unrelated text.

Runs on a pod set up by `runpod/olens_reads.sh setup` (vLLM image, merged lens on the volume):

    PYTHONPATH=olens/src python scripts/olens_residual_reads.py REQUESTS.jsonl OUT.jsonl MERGED_DIR [--layers 44,56]

REQUESTS.jsonl: scripts/olens_read_requests.py's format. Positions: stop tokens (that script's PUNCT, <|im_end|>,
</think>) inside an assistant message with n more tokens of the same message after them, up to --per-transcript per
transcript, evenly spaced. OUT.jsonl gets one line per (transcript, position, layer, config), appended per transcript;
finished transcripts are skipped on a rerun:

    {id, token_index, token, prefix (32 tokens up to p), continuation (the n tokens after p), layer, arm (main/wrong),
     beta, gamma, mu_scale, text, bullets [{text, fve, coef}], cont_fve (NNLS of z on the continuation atoms),
     joint_fve (bullets vs the real z), resid_fve (bullets vs z − fit of the arm), gain_fve (FVE of continuation atoms +
     bullets on z, minus cont_fve: what the bullets add beyond the continuation), cos_in (raw cos of h' with h)}
"""

import argparse
import itertools
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from olens.ar import AR
from olens.continuation import atoms, edit, nnls, whiten_atoms
from olens.lens import load_base
from olens.readout import Readout, bullets
from olens.vllm_lens import VllmLens
from olens_read_requests import PUNCT, SAMPLING, capture, render
from transformers import AutoTokenizer

MAIN = [("main", b, g, m) for b, g, m in itertools.product((0.0, 0.5, 1.0, 2.0), (1.0, 2.0, 4.0), (1.0, 0.5))]
WRONG = [("wrong", b, g, 1.0) for b, g in itertools.product((1.0, 2.0), (1.0, 2.0))]
CONFIGS = MAIN + WRONG
PREFIX = 32


def positions(tok, ids, where, msgs, n, k):
    """Stop tokens inside an assistant message with n more tokens of that message after them; k evenly spaced."""
    stops = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("</think>")}
    ok = [
        p
        for p in range(len(ids) - n)
        if (ids[p] in stops or PUNCT.match(tok.decode([ids[p]])))
        and where[p][0] >= 0
        and msgs[where[p][0]]["role"] == "assistant"
        and all(where[q][0] == where[p][0] for q in range(p + 1, p + 1 + n))
    ]
    return ok if len(ok) <= k else [ok[round(i * (len(ok) - 1) / (k - 1))] for i in range(k)]


@torch.no_grad()
def prepare(model, ar, readout, job, layers, n):
    """Capture h, fit the true and wrong continuations, build every config's h'. Returns the job with `items` (one per
    (position, layer, config)) and `vecs` [items, d] added."""
    ps, ids = job["positions"], job["ids"]
    acts = capture(model, ids, layers, ps)
    items, vecs = [], []
    for L in layers:
        W = readout.whitener(L)
        z = W(acts[L].to(W.W.device))
        A, cnt = atoms(ar, ids, ps, L, n)
        Dt = whiten_atoms(W, A, cnt)
        _, fit, cont_fve = nnls(z, Dt)
        wspans = job["wrong_spans"]
        Aw = torch.cat([ar.prefixes(wspans[i : i + 64], L) for i in range(0, len(wspans), 64)])
        Dw = whiten_atoms(W, Aw, torch.tensor([len(s) for s in wspans]))
        _, wfit, _ = nnls(z, Dw)
        wfit = wfit * (fit.norm(dim=-1, keepdim=True) / wfit.norm(dim=-1, keepdim=True).clamp_min(1e-9))
        job.setdefault("atoms", {})[L] = Dt
        for i, p in enumerate(ps):
            for arm, b, g, m in CONFIGS:
                f = fit[i] if arm == "main" else wfit[i]
                h = acts[L][i].to(W.W.device)
                hp = edit(W, h, f, b, g, m).float()
                cos_in = float(torch.cosine_similarity(hp, h.float(), dim=0))
                items.append({"i": i, "p": p, "L": L, "arm": arm, "beta": b, "gamma": g, "mu_scale": m,
                              "cont_fve": float(cont_fve[i]), "z": z[i], "r": z[i] - b * f, "cos_in": cos_in})  # fmt: skip
                vecs.append(hp.cpu())
    job["items"], job["vecs"] = items, torch.stack(vecs)
    return job


@torch.no_grad()
def score(tok, ar, readout, job, texts):
    """Per read: per-bullet FVE (cos² vs the real z, 0 if negative), joint/resid NNLS FVE, and the gain over the
    continuation atoms."""
    bls = [bullets(t) for t in texts]
    flat = [b for bl in bls for b in bl]
    spans = [tok(readout.lens.bullet_prefix + b, add_special_tokens=False).input_ids[:32] for b in flat]
    items = job["items"]
    lays = [items[k]["L"] for k, bl in enumerate(bls) for _ in bl]
    D = torch.cat([ar(spans[i : i + 256], lays[i : i + 256]) for i in range(0, len(spans), 256)]) if spans else torch.empty(0)
    out, j = [], 0
    for k, (it, bl) in enumerate(zip(items, bls)):
        rec = {"text": texts[k], "bullets": [], "joint_fve": 0.0, "resid_fve": 0.0, "gain_fve": 0.0}
        if bl:
            W = readout.whitener(it["L"])
            Db = W(D[j : j + len(bl)]).to(it["z"].dtype)
            j += len(bl)
            c, _, joint = nnls(it["z"][None], Db[None])
            _, _, resid = nnls(it["r"][None], Db[None])
            _, _, both = nnls(it["z"][None], torch.cat([job["atoms"][it["L"]][it["i"]], Db])[None])
            single = torch.cosine_similarity(Db, it["z"][None], dim=-1).clamp_min(0) ** 2
            bs = [{"text": b, "fve": round(float(single[x]), 4), "coef": round(float(c[0, x]), 4)} for x, b in enumerate(bl)]
            rec.update(bullets=sorted(bs, key=lambda x: -x["fve"]), joint_fve=round(float(joint[0]), 4),
                       resid_fve=round(float(resid[0]), 4), gain_fve=round(float(both[0]) - it["cont_fve"], 4))  # fmt: skip
        out.append(rec)
    return out


def write(tok, out, job, scored, n):
    ids, toks, row = job["ids"], job["toks"], job["row"]
    with out.open("a") as f:
        for it, s in zip(job["items"], scored):
            p = it["p"]
            rec = {"id": row["id"], "env": row.get("env"), "env_arm": row.get("arm"), "positive": row.get("positive"),
                   "token_index": p, "token": toks[p], "prefix": tok.decode(ids[max(0, p - PREFIX + 1) : p + 1]),
                   "continuation": tok.decode(ids[p + 1 : p + 1 + n]), "layer": it["L"], "arm": it["arm"],
                   "beta": it["beta"], "gamma": it["gamma"], "mu_scale": it["mu_scale"],
                   "cont_fve": round(it["cont_fve"], 4), "cos_in": round(it["cos_in"], 4), **s}  # fmt: skip
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("requests")
    ap.add_argument("out")
    ap.add_argument("merged")
    ap.add_argument("--layers", default="44,56")
    ap.add_argument("--n", type=int, default=32, help="continuation tokens; atoms are its prefixes 1..n")
    ap.add_argument("--per-transcript", type=int, default=50)
    ap.add_argument("--vllm-mem", type=float, default=0.55)
    ap.add_argument("--max-seqs", type=int, default=256)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    out = Path(args.out)
    done = {json.loads(line)["id"] for line in out.open()} if out.exists() else set()

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    rows = [json.loads(line) for line in Path(args.requests).read_text().splitlines() if line.strip()]
    jobs: list[dict] = []
    for row in rows:
        msgs, ids, where = render(tok, row)
        ps = positions(tok, ids, where, msgs, args.n, args.per_transcript)
        jobs.append({"row": row, "msgs": msgs, "ids": ids, "toks": [tok.decode([i]) for i in ids], "positions": ps})
    # wrong-continuation control: position i of transcript t gets a continuation from transcript t+1, same length
    rng = random.Random(0)
    for t, job in enumerate(jobs):
        other = jobs[(t + 1) % len(jobs)]
        job["wrong_spans"] = [other["ids"][q + 1 : q + 1 + args.n] for q in (rng.choice(other["positions"]) for _ in job["positions"])]
    jobs = [j for j in jobs if j["row"]["id"] not in done and j["positions"]]
    n_pos = sum(len(j["positions"]) for j in jobs)
    print(f"{len(jobs)} transcripts, {n_pos} positions x {len(layers)} layers x {len(CONFIGS)} configs", flush=True)

    lens = VllmLens(tok, args.merged, gpu_memory_utilization=args.vllm_mem, max_num_seqs=args.max_seqs)  # before the HF model
    _, model = load_base()
    lens.model = model
    ar = AR(model)
    readout = Readout(lens, ar)
    for L in layers:
        readout.whitener(L)

    # pipelined as in olens_read_requests.py: vLLM decodes job n in a thread while the main thread prepares n+1 and
    # scores n-1; all HF work stays on the main thread
    pool, t0, n_reads = ThreadPoolExecutor(1), time.time(), 0
    pending: dict | None = None
    for job in jobs + [None]:
        if job is not None:
            prepare(model, ar, readout, job, layers, args.n)
            job["fut"] = pool.submit(lens.read_items, job["vecs"], [it["L"] for it in job["items"]], SAMPLING)
        if pending is not None:
            texts = [t[0] for t in pending["fut"].result()]
            write(tok, out, pending, score(tok, ar, readout, pending, texts), args.n)
            n_reads += len(texts)
            rate = n_reads / (time.time() - t0)
            print(f"{pending['row']['id']}: {len(pending['positions'])} positions, {len(texts)} reads | {rate:.1f} reads/s", flush=True)
            pending.clear()
        pending = job


if __name__ == "__main__":
    main()
