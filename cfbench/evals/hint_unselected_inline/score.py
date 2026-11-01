"""Score hint_unselected_inline runs (the claim task): every draw one prediction (docs/cfbench-scoring.md).

    uv run python -m cfbench.evals.hint_unselected_inline.score data/runs/cfbench/<date>_<slug> [...]

Per condition: AUROC of p (knew vs didn't; the headline, chance 0.5) with a 95% bootstrap CI over cases (a case's draws
resampled together), % right at p >= 0.5 by class, and the paired bootstrap of the AUROC difference against the placebo and
transcript-only conditions (one-sided p). `stats(rows)` feeds the env viewer's claim table.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from cfbench.core import canon_condition
from cfbench.evals.hint_unselected_inline.eval import DIDNT, KNEW

BOOT = 2000
PAIRS = {f"olens_{m}_inline": (f"olens_{m}_placebo_inline", "transcript_inline") for m in ("prompt", "all")}


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a knew draw's p > a didn't draw's), ties half."""
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + eq / 2)


def by_case(rows: list[dict]) -> dict[str, dict[str, list[float]]]:
    """condition -> case item -> its draws' p."""
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[canon_condition(r["condition"])][r["item"]].append(float(r["p"]))
    return out


def _auroc_of(cases: dict[str, list[float]], cls: dict[str, bool], items: list[str]) -> float:
    pos = np.array([p for i in items if cls[i] for p in cases[i]])
    neg = np.array([p for i in items if not cls[i] for p in cases[i]])
    return auroc(pos, neg)


def stats(rows: list[dict]) -> list[dict]:
    cls = {r["item"]: r["model"] == KNEW for r in rows}
    assert {r["model"] for r in rows} <= {KNEW, DIDNT}
    bc = by_case(rows)
    rng = np.random.default_rng(0)
    out = []
    for cond, cases in sorted(bc.items()):
        items = sorted(cases)
        boots = []
        for _ in range(BOOT):
            samp = [items[k] for k in rng.integers(len(items), size=len(items))]
            boots.append(_auroc_of(cases, cls, samp))
        right = {c: [float((p >= 0.5) == cls[i]) for i in items if cls[i] == c for p in cases[i]] for c in (True, False)}
        out.append({"condition": cond, "n_cases": len(items), "n_draws": sum(map(len, cases.values())),
                    "auroc": _auroc_of(cases, cls, items), "ci": [float(v) for v in np.percentile(boots, [2.5, 97.5])],
                    "accuracy": float(np.mean(right[True] + right[False])), "acc_knew": float(np.mean(right[True])),
                    "acc_didnt": float(np.mean(right[False]))})  # fmt: skip
    return out


def paired(rows: list[dict]) -> list[dict]:
    """Paired bootstrap over cases of AUROC(a) − AUROC(b); one-sided p = share of resamples with a ≤ b."""
    cls = {r["item"]: r["model"] == KNEW for r in rows}
    bc = by_case(rows)
    rng = np.random.default_rng(1)
    out = []
    for a, others in PAIRS.items():
        for b in others:
            if a not in bc or b not in bc:
                continue
            items = sorted(set(bc[a]) & set(bc[b]))
            d = _auroc_of(bc[a], cls, items) - _auroc_of(bc[b], cls, items)
            boots = []
            for _ in range(BOOT):
                samp = [items[k] for k in rng.integers(len(items), size=len(items))]
                boots.append(_auroc_of(bc[a], cls, samp) - _auroc_of(bc[b], cls, samp))
            out.append({"a": a, "b": b, "n": len(items), "diff": d, "ci": [float(v) for v in np.percentile(boots, [2.5, 97.5])],
                        "p_one_sided": float(np.mean(np.array(boots) <= 0))})  # fmt: skip
    return out


def main():
    rows = [json.loads(line) for path in sys.argv[1:] for line in (Path(path) / "predictions.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r.get("p") is not None]
    for s in stats(rows):
        lo, hi = s["ci"]
        print(f"{s['condition']:24s} n={s['n_cases']:4d} draws={s['n_draws']:5d}  AUROC {s['auroc']:.3f} [{lo:.3f}, {hi:.3f}]"
              f"  %right {100 * s['accuracy']:.1f} (knew {100 * s['acc_knew']:.1f}, didn't {100 * s['acc_didnt']:.1f})")  # fmt: skip
    for x in paired(rows):
        print(f"{x['a']} − {x['b']}: ΔAUROC {x['diff']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}] p={x['p_one_sided']:.4f} (n={x['n']})")


if __name__ == "__main__":
    main()
