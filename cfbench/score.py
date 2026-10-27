"""Score cfbench predictions: RMSE of the predicted rate (docs/cfbench-scoring.md).

Per case (an ordered arm pair), the squared error (p_hat - k_cf/n_cf)^2 averaged over draws; per predictor and condition,
RMSE over cases with equal weight, next to the RMSE of "the edit does nothing" (predict the base arm's rate), and the
Brier skill 1 - sum MSE / sum MSE_null (0 = the null, 1 = perfect, negative = worse than the null).

The older binomial log-likelihood skill (numbers from before 2026-10-26) is no longer printed here; the report package
keeps it (`cfbench.report.data.log_skill`) for comparison with older numbers.

    uv run python -m cfbench.score data/runs/cfbench/<run dir> [...]   # or the pre-runs.py TAG.jsonl files
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    paths = [p / "predictions.jsonl" if p.is_dir() else p for p in map(Path, sys.argv[1:])]  # run dirs or .jsonl files
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
    cases = defaultdict(list)  # (predictor, case) -> rows
    for r in rows:
        cases[(r.get("condition", "transcript"), r["predictor"], r["env"], r["model"], r["base_arm"], r["cf_arm"])].append(r)

    totals = defaultdict(lambda: [0.0, 0.0, 0])  # (condition, predictor) -> [mse, mse_null, cases]
    print(
        f"{'condition':13} {'predictor':30} {'case':38} {'base':>7} {'cf':>7} {'p_hat (each draw)':28} "
        f"{'invalid':>7} {'rmse':>5} {'null':>5}"
    )
    for (condition, predictor, env, model, base_arm, cf_arm), group in sorted(cases.items()):
        r0 = group[0]
        k, n = r0["k_cf"], r0["n_cf"]
        valid = [r for r in group if r["valid"]]
        if not valid:
            continue
        q, q_base = k / n, r0["k_base"] / r0["n_base"]
        mse = float(np.mean([(r["p"] - q) ** 2 for r in valid]))
        mse_null = (q_base - q) ** 2
        t = totals[condition, predictor]
        t[0] += mse
        t[1] += mse_null
        t[2] += 1
        ps = " ".join(f"{r['p']:.2f}" for r in sorted(valid, key=lambda r: r["draw"]))
        case = f"{env}/{model.split('/')[-1]} {base_arm}->{cf_arm}"
        print(
            f"{condition:13} {predictor:30} {case:38} {r0['k_base']:>3}/{r0['n_base']:<3} {k:>3}/{n:<3} {ps:28} "
            f"{len(group) - len(valid):>7} {np.sqrt(mse) * 100:>4.0f}pp {np.sqrt(mse_null) * 100:>3.0f}pp"
        )
    print()
    for (condition, predictor), (mse, mse_null, n_cases) in sorted(totals.items()):
        print(
            f"{condition:13} {predictor:30} RMSE {np.sqrt(mse / n_cases) * 100:.1f}pp "
            f"(edit does nothing: {np.sqrt(mse_null / n_cases) * 100:.1f}pp), Brier skill {1 - mse / mse_null:.3f}, {n_cases} cases"
        )


if __name__ == "__main__":
    main()
