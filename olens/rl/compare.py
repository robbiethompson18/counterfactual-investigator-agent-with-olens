"""Side by side: two lenses' reads at the same held-out positions (reads_<tag>.jsonl from train.py eval), for reading the text
rather than the reward.

    python olens/rl/compare.py BASE.jsonl TRAINED.jsonl [--n 25] [--seed 0] [--layers 20,24]
"""

import argparse
import json
import random
from pathlib import Path


def load(p):
    return {r["idx"]: r for r in map(json.loads, Path(p).read_text().splitlines())}


def show(read):
    return f"[marg {read['marginal']:.2f} plain {read['fve_plain']:.2f}] " + " | ".join(b[:90] for b in read["bullets"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reads", type=int, default=2, help="reads shown per lens per position")
    ap.add_argument("--layers", help="comma-separated subset")
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    idx = sorted(set(A) & set(B))
    if args.layers:
        keep = {int(x) for x in args.layers.split(",")}
        idx = [i for i in idx if A[i]["layer"] in keep]
    for i in random.Random(args.seed).sample(idx, min(args.n, len(idx))):
        a, b = A[i], B[i]
        print("=" * 120)
        print(f"#{i} L{a['layer']}  context: ...{a['context'][-160:]!r}")
        print(f"   next text: {a['continuation'][:160]!r}")
        for r in a["reads"][: args.reads]:
            print("   A " + show(r))
        for r in b["reads"][: args.reads]:
            print("   B " + show(r))


if __name__ == "__main__":
    main()
