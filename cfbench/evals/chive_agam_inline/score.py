"""Score chive_agam_inline runs: AUROC of P(claim true) per condition, and paired differences by claim.

    uv run python -m cfbench.evals.chive_agam_inline.score RUN [RUN ...]   # e.g. the no-lens and the lens runs

Each draw is one prediction, as in CHIVE. CIs (95%) bootstrap claims, every condition on the same resampled claims.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from cfbench.evals.chive_agam_inline.eval import CONDITIONS
from cfbench.evals.chive_inline.score import auroc

BOOT = 1000
PAIRS = [("transcript_inline", "no_transcript_inline"), ("olens_inline", "transcript_inline"), ("olens_inline", "olens_placebo_inline"),
         ("nla_inline", "transcript_inline"), ("nla_inline", "nla_placebo_inline"), ("jlens_inline", "transcript_inline"),
         ("jlens_inline", "jlens_placebo_inline")]  # fmt: skip


def table(runs: list[Path]) -> tuple[list[str], dict[str, float], np.ndarray, int]:
    """(conditions present, point AUROC per condition, bootstrap AUROCs [BOOT x conditions], claims)."""
    claims: dict = defaultdict(lambda: [0, defaultdict(list)])
    for r in (json.loads(line) for run in runs for line in (run / "predictions.jsonl").open()):
        if r.get("p") is not None:
            claims[r["item"]][0] = r["k_cf"]
            claims[r["item"]][1][r["condition"]].append(r["p"])
    conds = [c for c in CONDITIONS if any(c in v[1] for v in claims.values())]
    ids = list(claims)

    def aurocs(sample: list[str]) -> list[float]:
        return [
            auroc(np.array([claims[i][0] for i in sample for _ in claims[i][1][c]]), np.array([p for i in sample for p in claims[i][1][c]]))
            for c in conds
        ]

    rng = np.random.default_rng(0)
    boots = np.array([aurocs([ids[i] for i in rng.integers(len(ids), size=len(ids))]) for _ in range(BOOT)])
    return conds, dict(zip(conds, aurocs(ids), strict=True)), boots, len(ids)


def main(runs: list[Path]):
    conds, point, boots, n = table(runs)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    print(f"{n} claims")
    for k, c in enumerate(conds):
        print(f"  {c:24s} AUROC {point[c]:.3f} [{lo[k]:.3f}, {hi[k]:.3f}]")
    for a, b in PAIRS:
        if a in conds and b in conds:
            d = boots[:, conds.index(a)] - boots[:, conds.index(b)]
            print(f"  {a} - {b}: {point[a] - point[b]:+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")


if __name__ == "__main__":
    main([Path(a) for a in sys.argv[1:]])
