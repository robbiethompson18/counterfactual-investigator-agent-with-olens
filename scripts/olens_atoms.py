"""Atoms for an existing O-lens read run (scripts/olens_read_requests.py output), so scripts/olens_novelty.py can score
it on CPU. No lens decoding: the bullets are the run's own. Per transcript, at up to --per-transcript of its read
positions (evenly spaced, any role), it saves the whitened activation z, the whitened AR atoms of the continuation's
prefixes (--prefixes of the next 32 tokens; a prefix past the transcript's end is a zero row), and the whitened AR atoms
of the read's bullets (the lens's bullet prefix + text, as the run scored them).

Runs on a pod set up by `runpod/olens_reads.sh setup` (needs the HF base + AR + whiteners, not the merged lens):

    PYTHONPATH=olens/src python scripts/olens_atoms.py RUN_DIR OUT_DIR [--layer 44] [--per-transcript 50]

RUN_DIR holds requests.jsonl, reads.jsonl and meta.json (params.lens picks the bullet prefix). OUT_DIR/atoms/<id with
"/" -> "__">.pt, fp16, the residual-reads format (scripts/olens_residual_reads.py save_atoms) plus labels and text:
{id, env, arm, positive, lens, positions, layers, prefix_lengths, z {L: [P, d]}, cont {L: [P, n, d]},
plain_bullets {L: [P] texts}, plain_atoms {L: [P] [k, d]}, text {token, role, prefix, continuation: [P]}}.
Transcripts with a file already are skipped, so a rerun resumes.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from olens.ar import AR
from olens.fve import Whitener
from olens.lens import load_base
from olens_read_requests import capture, render

BULLET_PREFIX = {"olens_s3d_space_rl600": " ", "olens_s3d_rl600": ""}  # olens.lens.bullet_prefix, by adapter name
N = 32  # continuation tokens; the AR reads spans of at most 32


def jsonl(path):
    with open(path) as f:
        yield from map(json.loads, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--layer", type=int, help="default: 44 if the run read it, else 48")
    ap.add_argument("--per-transcript", type=int, default=50)
    ap.add_argument("--prefixes", default="1,2,4,8,16,32")
    args = ap.parse_args()
    lens = json.loads((args.run / "meta.json").read_text())["params"]["lens"]
    prefix = BULLET_PREFIX[lens]
    lengths = [int(k) for k in args.prefixes.split(",")]
    layers_read = {r["layer"] for r in jsonl(args.run / "reads.jsonl")}
    L = args.layer or (44 if 44 in layers_read else 48)
    assert L in layers_read, f"layer {L} not in the run's layers {sorted(layers_read)}"
    reads = defaultdict(dict)  # id -> {token_index: read row at L}
    for r in jsonl(args.run / "reads.jsonl"):
        if r["layer"] == L:
            reads[r["id"]][r["token_index"]] = r
    requests = {r["id"]: r for r in jsonl(args.run / "requests.jsonl")}
    (args.out / "atoms").mkdir(parents=True, exist_ok=True)
    todo = [t for t in sorted(reads) if not (args.out / "atoms" / (t.replace("/", "__") + ".pt")).exists()]
    print(f"{len(todo)} of {len(reads)} transcripts to do at L{L}, lens {lens} (bullet prefix {prefix!r})", flush=True)

    tok, model = load_base()
    ar = AR(model)
    W = Whitener(L, device=model.device, dtype=torch.float32)
    bad = 0
    for n, tid in enumerate(todo):
        _, ids, _ = render(tok, requests[tid])
        rows = reads[tid]
        if max(rows) >= len(ids) or any(tok.decode([ids[p]]) != r["token"] for p, r in rows.items()):
            bad += 1  # rendering differs from the run's (template or request changed): skip rather than misalign
            print(f"{tid}: tokens don't match the run's reads, skipped", flush=True)
            continue
        ps = [p for p in sorted(rows) if p < len(ids) - 1]  # the last token has no continuation
        ps = [ps[i] for i in sorted(set(np.linspace(0, len(ps) - 1, min(args.per_transcript, len(ps))).round().astype(int)))]
        h = capture(model, ids, [L], ps)[L]
        spans = [ids[p + 1 : p + 1 + N] for p in ps]
        A = torch.zeros(len(ps), len(lengths), h.shape[1], device=model.device)
        for i in range(0, len(spans), 64):
            a = ar.prefixes(spans[i : i + 64], L)  # [b, longest span, d]
            for j, k in enumerate(lengths):
                if k <= a.shape[1]:
                    A[i : i + len(a), j] = a[:, k - 1]  # zero for spans shorter than k (prefixes() zeroes past the end)
        texts = [[b["text"] for b in rows[p]["bullets"]] for p in ps]
        bspans = [tok(prefix + t, add_special_tokens=False).input_ids[:32] for bl in texts for t in bl]
        D = torch.cat([ar(bspans[i : i + 256], L) for i in range(0, len(bspans), 256)]) if bspans else torch.zeros(0, h.shape[1])
        D = W(D).half().cpu() if len(D) else D
        cuts = np.cumsum([0] + [len(bl) for bl in texts])
        zero = (A != 0).any(-1, keepdim=True)  # keep padded prefixes at exactly 0 after whitening
        row = requests[tid]
        torch.save({"id": tid, "env": row.get("env"), "arm": row.get("arm"), "positive": row.get("positive"), "lens": lens,
                    "positions": ps, "layers": [L], "prefix_lengths": lengths,
                    "z": {L: W(h.to(model.device)).half().cpu()}, "cont": {L: (W(A) * zero).half().cpu()},
                    "plain_bullets": {L: texts}, "plain_atoms": {L: [D[cuts[i] : cuts[i + 1]] for i in range(len(ps))]},
                    "text": {"token": [rows[p]["token"] for p in ps], "role": [rows[p]["role"] for p in ps],
                             "prefix": [rows[p]["prefix"] for p in ps], "continuation": [tok.decode(s) for s in spans]}},
                   args.out / "atoms" / (tid.replace("/", "__") + ".pt"))  # fmt: skip
        print(f"[{n + 1}/{len(todo)}] {tid}: {len(ids)} tokens, {len(ps)} positions, {len(bspans)} bullets", flush=True)
    print(f"done; {bad} transcripts skipped for token mismatches", flush=True)


if __name__ == "__main__":
    main()
