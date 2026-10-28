"""Parity between two DSV4 O-lens runs (read.py / vllm_read.py / vllm_capture_read.py outputs): greedy-prefix agreement of
the reads on the same (id, pos, layer), and agreement of the first-reply logits (accept minus reject log-prob) per id.

    python compare_reads.py RUN_A RUN_B [--reads reads_L30.jsonl]   # dirs holding reads_L*.jsonl + logits.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--reads", default="reads_L30.jsonl")
args = ap.parse_args()
A, B = Path(args.a), Path(args.b)

ra = {(x["id"], x["pos"], x["layer"]): x for x in rows(A / args.reads)}
rb = {(x["id"], x["pos"], x["layer"]): x for x in rows(B / args.reads)}
pairs = [(ra[k], rb[k]) for k in ra if k in rb]
print(f"reads: {len(ra)} in A, {len(rb)} in B, {len(pairs)} shared")
if pairs:
    same = lambda n: np.mean([x["greedy"].strip()[:n] == y["greedy"].strip()[:n] for x, y in pairs])
    print(f"  greedy first 10 chars equal {same(10):.3f}, 20 {same(20):.3f}, 40 {same(40):.3f}, whole {same(10**6):.3f}")
    tok_same = np.mean([x["token"] == y["token"] for x, y in pairs])
    print(f"  same token text at shared keys {tok_same:.3f} (1.0 = identical positions)")
    for x, y in pairs[:5]:
        print(f"  {x['token']!r}\n    A {x['greedy'][:80]!r}\n    B {y['greedy'][:80]!r}")

la = {x["id"]: x for x in rows(A / "logits.jsonl")}
lb = {x["id"]: x for x in rows(B / "logits.jsonl")}
ids = [i for i in la if i in lb]
print(f"logits: {len(la)} in A, {len(lb)} in B, {len(ids)} shared")
if len(ids) > 1:
    da = np.array([la[i]["accept"] - la[i]["reject"] for i in ids])
    db = np.array([lb[i]["accept"] - lb[i]["reject"] for i in ids])
    for name in ("reject", "accept"):
        va, vb = (np.array([d[i][name] for i in ids]) for d in (la, lb))
        corr = np.corrcoef(va, vb)[0, 1]
        print(
            f"  log P({name}): corr {corr:.4f}, mean |diff| {np.abs(va - vb).mean():.3f} nats (A mean {va.mean():.2f}, B {vb.mean():.2f})"
        )
    corr, sign = np.corrcoef(da, db)[0, 1], np.mean(np.sign(da) == np.sign(db))
    print(f"  accept - reject: corr {corr:.4f}, mean |diff| {np.abs(da - db).mean():.3f}, sign agreement {sign:.3f}")
    top_same = np.mean([la[i]["top_first"][0][0] == lb[i]["top_first"][0][0] for i in ids])
    print(f"  same top-1 first reply token {top_same:.3f}")
