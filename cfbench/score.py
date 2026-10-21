"""Binomial log-likelihood skill of cfbench predictions.

For a case with k_cf behaviors in n_cf counterfactual-arm runs, LL(p) = log Binom(k_cf; n_cf, p). Per predictor:

    skill = sum_cases [mean_draws LL(p_hat) - LL(null)] / sum_cases [LL(oracle) - LL(null)]

null = smoothed base-arm rate (the edit does nothing); oracle = observed counterfactual rate. Both are clipped to the same
[0.01, 0.99] the predictor is held to, so the oracle is the best allowed answer: skill 0 = no better than "nothing changes",
1 = as good as knowing the rate, negative = worse than ignoring the edit. Draws are averaged within a case so cases count once.

    uv run python -m cfbench.score data/runs/cfbench/TAG.jsonl [...]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binom

from cfbench.predict import P_MAX, P_MIN


def ll(k: int, n: int, p: float) -> float:
    return float(binom.logpmf(k, n, np.clip(p, P_MIN, P_MAX)))


def main():
    rows = [json.loads(line) for path in sys.argv[1:] for line in Path(path).read_text().splitlines()]
    cases = defaultdict(list)  # (predictor, case) -> rows
    for r in rows:
        cases[(r["predictor"], r["env"], r["model"], r["base_arm"], r["cf_arm"])].append(r)

    totals = defaultdict(lambda: [0.0, 0.0])  # predictor -> [agent gain over null, oracle gain over null]
    print(f"{'predictor':34} {'case':38} {'base':>7} {'cf':>7} {'p_hat (each draw)':28} {'invalid':>7} {'skill':>6}")
    for (predictor, env, model, base_arm, cf_arm), group in sorted(cases.items()):
        r0 = group[0]
        k, n = r0["k_cf"], r0["n_cf"]
        null = ll(k, n, (r0["k_base"] + 0.5) / (r0["n_base"] + 1))
        oracle = ll(k, n, k / n)
        valid = [r for r in group if r["valid"]]
        gain = float(np.mean([ll(k, n, r["p"]) for r in valid])) - null
        totals[predictor][0] += gain
        totals[predictor][1] += oracle - null
        ps = " ".join(f"{r['p']:.2f}" for r in sorted(valid, key=lambda r: r["draw"]))
        case = f"{env}/{model.split('/')[-1]} {base_arm}->{cf_arm}"
        print(
            f"{predictor:34} {case:38} {r0['k_base']:>3}/{r0['n_base']:<3} {k:>3}/{n:<3} {ps:28} "
            f"{len(group) - len(valid):>7} {gain / (oracle - null):>6.2f}"
        )
    print()
    for predictor, (gain, oracle_gain) in sorted(totals.items()):
        print(f"{predictor:34} skill {gain / oracle_gain:.3f}  (gain over null {gain:.2f} nats, oracle {oracle_gain:.2f})")


if __name__ == "__main__":
    main()
