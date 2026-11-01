"""The hinted-MCQ pool with no second-model filter (Robbie, 2026-10-31: "not filtering on disagreement").

Items: every MMLU-Pro test question Qwen3.6-27B (thinking off) answers consistently with no hint (>= 7 of 8 samples agree,
the 2026-10-28_all12k-nohint census over all 12,032 questions), minus the 2,000-question probe half of the mcq_unselected
split (docs/mcq-unselected.md), so a probe can be trained there later with no leakage. Class **didn't**: the consistent
answer is wrong; **knew**: it is right. Nothing about DeepSeek or any other model.

`pool()` is every didn't item plus, per gold letter, a seeded sample of twice as many knew items; the hinted rollouts ran
over this pool, then over the 294 knew items eval.select() drew outside it (its own seed; the top-up run in eval.BASE_RUNS).
`prep` then applies the hint-follow filter and balances the classes within each gold letter.

    uv run python -m cfbench.evals.hint_unselected_inline.items    # prints the pool as a comma list for `environments run --items`
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

RUNS = Path("data/runs")
CENSUS = Path("data/envs/hint_mcq/2026-10-28_all12k-nohint/census.csv")
SPLIT = RUNS / "cfbench_evals/2026-10-29_mcq-unselected-inputs/split.json"
KNEW_PER_DIDNT = 2
SEED = 0


def census() -> dict[str, dict]:
    """item -> census row (consistent, not in the probe half)."""
    probe = set(map(str, json.load(SPLIT.open())["probe"]))
    rows = {r["item"]: r for r in csv.DictReader(CENSUS.open()) if r["consistent_80"] == "True" and r["item"] not in probe}
    for r in rows.values():
        r["knew"] = r["modal_correct"] == "True"
    return rows


def pool() -> list[str]:
    rows = census()
    by = defaultdict(lambda: defaultdict(list))
    for i, r in rows.items():
        by[r["gold"]][r["knew"]].append(i)
    out = []
    for gold in sorted(by):
        didnt, knew = sorted(by[gold][False]), sorted(by[gold][True])
        out += didnt + random.Random(f"{SEED}:{gold}").sample(knew, min(len(knew), KNEW_PER_DIDNT * len(didnt)))
    return out


if __name__ == "__main__":
    print(",".join(pool()))
