"""Score cfbench predictions: RMSE of the predicted rate (the standard, docs/cfbench-scoring.md) and the older log-likelihood skill.

Per case (an ordered arm pair), the squared error (p_hat - k_cf/n_cf)^2 averaged over draws; per predictor and condition,
RMSE over cases with equal weight, next to the RMSE of "the edit does nothing" (predict the base arm's rate), and the
Brier skill 1 - sum MSE / sum MSE_null (0 = the null, 1 = perfect, negative = worse than the null). The math is
`cfbench.report.data.Case`, shared with the report charts.

Also printed, for comparison with numbers from before 2026-10-26: binomial log-likelihood skill,
sum_cases [mean_draws LL(p_hat) - LL(null)] / sum_cases [LL(oracle) - LL(null)], with the smoothed base rate as null and
the observed counterfactual rate as oracle, both clipped to the predictor's [0.01, 0.99]. It weights cases by oracle gain
(so by n_cf and effect size), which is why it was replaced.

    uv run python -m cfbench.score data/runs/cfbench/<run dir> [...]   # or the pre-runs.py TAG.jsonl files
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cfbench.report.data import Case, brier_skill, log_skill, rmse, rmse_null

Group = tuple[str, str]  # (condition, predictor)
CaseKey = tuple[str, str, str, str]  # (env, target model, base_arm, cf_arm)


def load(paths: list[Path]) -> tuple[dict[Group, dict[CaseKey, Case]], Counter]:
    """Cases per (condition, predictor) from the valid rows of each predictions file (p in draw order), and the invalid-row
    count per (condition, predictor, *case). Rows from before the condition field are `transcript`."""
    cases: dict[Group, dict[CaseKey, Case]] = defaultdict(dict)
    invalid: Counter = Counter()
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
    for r in sorted(rows, key=lambda r: r["draw"]):
        condition = r.get("condition", "transcript")
        group, key = (condition, r["predictor"]), (r["env"], r["model"], r["base_arm"], r["cf_arm"])
        if not r["valid"]:
            invalid[group + key] += 1
            continue
        case = cases[group].setdefault(key, Case(r["env"], r["base_arm"], r["cf_arm"], r["k_base"], r["n_base"], r["k_cf"], r["n_cf"]))
        case.p.setdefault(condition, []).append(r["p"])
    return cases, invalid


def main():
    paths = [p / "predictions.jsonl" if p.is_dir() else p for p in map(Path, sys.argv[1:])]  # run dirs or .jsonl files
    cases, invalid = load(paths)
    print(
        f"{'condition':13} {'predictor':30} {'case':38} {'base':>7} {'cf':>7} {'p_hat (each draw)':28} {'invalid':>7} "
        f"{'rmse':>5} {'null':>5} gain/oracle"
    )
    for (condition, predictor), group in sorted(cases.items()):
        for (env, model, base_arm, cf_arm), c in sorted(group.items()):
            ps = " ".join(f"{p:.2f}" for p in c.p[condition])
            name = f"{env}/{model.split('/')[-1]} {base_arm}->{cf_arm}"
            print(
                f"{condition:13} {predictor:30} {name:38} {c.k_base:>3}/{c.n_base:<3} {c.k_cf:>3}/{c.n_cf:<3} {ps:28} "
                f"{invalid[condition, predictor, env, model, base_arm, cf_arm]:>7} {np.sqrt(c.mse(condition)) * 100:>4.0f}pp "
                f"{np.sqrt(c.mse_null()) * 100:>3.0f}pp {c.log_gain(condition):>7.1f}/{c.log_oracle():<6.1f}"
            )
    print()
    for (condition, predictor), group in sorted(cases.items()):
        print(
            f"{condition:13} {predictor:30} RMSE {rmse(group, condition) * 100:.1f}pp "
            f"(edit does nothing: {rmse_null(group) * 100:.1f}pp), Brier skill {brier_skill(group, condition):.3f}, "
            f"{len(group)} cases  |  log skill {log_skill(group, condition):.3f}"
        )


if __name__ == "__main__":
    main()
