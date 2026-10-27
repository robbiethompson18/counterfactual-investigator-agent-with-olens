"""Paired comparison of cfbench conditions over many runs: pooled skill per condition, the skill difference of one condition
against another with a 95% case bootstrap, and a per-env table with O-lens tool use.

    uv run python scripts/cfbench_compare.py 'data/runs/cfbench/2026-10-25_nocot-cf-*' [--vs olens:agentic --vs olens:olens_placebo]

Skill is cfbench/score.py's: per case, mean over draws of the binomial log-likelihood of the counterfactual arm's k/n at the
predicted p, minus the "edit does nothing" null (the smoothed base rate), normalized by the oracle (the observed rate). A
case is (env, base arm, cf arm); runs that share a case are pooled (e.g. a `--skip` fill-in run). The bootstrap resamples
cases, which is the unit the pre-registration counts, not draws.
"""

import argparse
import collections
import glob
import json
import random
from math import lgamma, log
from pathlib import Path

P_MIN, P_MAX = 0.01, 0.99


def ll(k, n, p):
    p = min(max(p, P_MIN), P_MAX)
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1) + k * log(p) + (n - k) * log(1 - p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", help="glob of run dirs")
    ap.add_argument("--vs", action="append", default=[], help="A:B pairs to compare (default olens:agentic, olens:olens_placebo)")
    ap.add_argument("--boots", type=int, default=5000)
    args = ap.parse_args()
    pairs = [v.split(":") for v in args.vs] or [["olens", "agentic"], ["olens", "olens_placebo"]]

    rows = [
        json.loads(line) for d in glob.glob(args.runs) for line in Path(d, "predictions.jsonl").read_text().splitlines() if line.strip()
    ]
    cases = collections.defaultdict(lambda: collections.defaultdict(list))
    meta = {}
    for r in rows:
        c = (r["env"].split(":")[1].split("/")[0], r["base_arm"], r["cf_arm"])
        meta[c] = (r["k_cf"], r["n_cf"], r["k_base"], r["n_base"])
        if r.get("valid"):
            cases[c][r["condition"]].append(r)
    conds = sorted({r["condition"] for r in rows})

    def gain(c, cond):  # (LL gain over the null, averaged over draws; oracle gain over the null)
        k, n, kb, nb = meta[c]
        null = ll(k, n, (kb + 0.5) / (nb + 1))
        return sum(ll(k, n, r["p"]) for r in cases[c][cond]) / len(cases[c][cond]) - null, ll(k, n, k / n) - null

    full = [c for c in cases if all(cases[c].get(x) for x in conds)]
    print(f"{len(rows)} predictions, {len(cases)} cases, {len(full)} with every condition ({', '.join(conds)})")
    for cond in conds:
        print(f"  skill {cond:14s} {sum(gain(c, cond)[0] for c in full) / sum(gain(c, cond)[1] for c in full):+.3f}")

    def diff(sample, a, b):
        return sum(gain(c, a)[0] - gain(c, b)[0] for c in sample) / sum(gain(c, a)[1] for c in sample)

    rng = random.Random(0)
    for a, b in pairs:
        boots = sorted(diff([rng.choice(full) for _ in full], a, b) for _ in range(args.boots))
        lo, hi = boots[int(0.025 * args.boots)], boots[int(0.975 * args.boots) - 1]
        print(f"  {a} - {b}: {diff(full, a, b):+.3f}  95% case bootstrap [{lo:+.3f}, {hi:+.3f}]")

    print(f"\nper env: cases, skill per condition ({' / '.join(conds)}), O-lens calls per olens prediction and how many made none")
    by_env = collections.defaultdict(list)
    for c in full:
        by_env[c[0]].append(c)
    for env, cs in sorted(by_env.items()):
        skills = " / ".join(f"{sum(gain(c, cond)[0] for c in cs) / sum(gain(c, cond)[1] for c in cs):+.2f}" for cond in conds)
        lens = [sum(v for k, v in (r["tool_calls"] or {}).items() if "olens" in k) for c in cs for r in cases[c].get("olens", [])]
        use = f"lens {sum(lens) / len(lens):.1f}, {sum(x == 0 for x in lens)}/{len(lens)} zero" if lens else ""
        print(f"  {env:34s} {len(cs):2d}  {skills}   {use}")


if __name__ == "__main__":
    main()
