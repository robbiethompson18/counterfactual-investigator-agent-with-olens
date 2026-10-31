"""Score chive_inline runs: AUROC of P(claim true) per target and condition, beside CHIVE's Opus 4.8 plain / cf arms.

    uv run python -m cfbench.evals.chive_inline.score data/runs/cfbench/<date>_<slug>

Each draw is one prediction, as each of CHIVE's rows is, so AUROC pools every draw of every claim. CIs (95%) bootstrap
claims; "transcript - edit" is the paired difference on the same resampled claims.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from cfbench.evals.chive_inline.eval import CONDITIONS
from chive_critique.data import ADAM_LABELS, adam

BOOT = 1000


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    r, pos = rankdata(p), y.sum()
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos)))


def opus(version: str, arm: str) -> float:
    rows = [r for r in adam(f"{version}/binary_evals/{arm}_claude-opus-4-8_high_conc3.jsonl") if r["prob_true"] not in (None, "None")]
    return auroc(np.array([str(r["is_true"]) == "True" for r in rows]), np.array([float(r["prob_true"]) for r in rows]))


def main(run: Path):
    preds = defaultdict(lambda: defaultdict(list))  # version -> claim -> [(condition, p, y)]
    for r in map(json.loads, (run / "predictions.jsonl").read_text().splitlines()):
        if r.get("p") is not None:
            preds[r["version"]][r["item"]].append((r["condition"], r["p"], r["k_cf"]))
    rng = np.random.default_rng(0)
    for version, claims in preds.items():
        ids = list(claims)

        def aurocs(sample: list[str], claims=claims) -> list[float]:
            rows = [x for i in sample for x in claims[i]]
            return [
                auroc(np.array([y for c, _, y in rows if c == cond]), np.array([p for c, p, _ in rows if c == cond])) for cond in CONDITIONS
            ]

        point = aurocs(ids)
        boots = np.array([aurocs([ids[i] for i in rng.integers(len(ids), size=len(ids))]) for _ in range(BOOT)])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        dlo, dhi = np.percentile(boots[:, 0] - boots[:, 1], [2.5, 97.5])
        n = {cond: sum(c == cond for i in ids for c, _, _ in claims[i]) for cond in CONDITIONS}
        print(f"\n{ADAM_LABELS[version]} ({len(ids)} claims)")
        for cond, a, l, h in zip(CONDITIONS, point, lo, hi):
            print(f"  Haiku {cond:22s} AUROC {a:.3f} [{l:.3f}, {h:.3f}]  ({n[cond]} predictions)")
        print(f"  transcript - edit only      {point[0] - point[1]:+.3f} [{dlo:+.3f}, {dhi:+.3f}]")
        print(f"  CHIVE Opus 4.8 plain {opus(version, 'plain'):.3f}, cf {opus(version, 'cf'):.3f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
