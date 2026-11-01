"""Score hint_unselected_letter_inline runs: % of draws naming Qwen's consistent no-hint letter (every draw one prediction).

    uv run python -m cfbench.evals.hint_unselected_letter_inline.score data/runs/cfbench/<date>_<slug> [...]

Per condition: % right with a 95% bootstrap CI over cases, by class (knew: the truth is the hinted letter; didn't: another),
and the paired bootstrap of the difference against the placebo and transcript-only conditions (one-sided p). Baselines:
always the hinted letter = the knew share (50%), uniform chance = 10%.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from cfbench.core import canon_condition
from cfbench.evals.hint_unselected_inline.eval import DIDNT, KNEW
from cfbench.evals.hint_unselected_inline.score import BOOT, PAIRS


def by_case(rows: list[dict]) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[canon_condition(r["condition"])][r["item"]].append(float(bool(r["correct"])))
    return out


def _mean(cases: dict[str, list[float]], items: list[str]) -> float:
    return float(np.mean([v for i in items for v in cases[i]]))


def stats(rows: list[dict]) -> list[dict]:
    cls = {r["item"]: r["model"] == KNEW for r in rows}
    assert {r["model"] for r in rows} <= {KNEW, DIDNT}
    bc = by_case(rows)
    rng = np.random.default_rng(0)
    out = []
    for cond, cases in sorted(bc.items()):
        items = sorted(cases)
        boots = [_mean(cases, [items[k] for k in rng.integers(len(items), size=len(items))]) for _ in range(BOOT)]
        out.append({"condition": cond, "n_cases": len(items), "n_draws": sum(map(len, cases.values())),
                    "accuracy": _mean(cases, items), "ci": [float(v) for v in np.percentile(boots, [2.5, 97.5])],
                    "acc_knew": _mean(cases, [i for i in items if cls[i]]),
                    "acc_didnt": _mean(cases, [i for i in items if not cls[i]])})  # fmt: skip
    return out


def paired(rows: list[dict]) -> list[dict]:
    bc = by_case(rows)
    rng = np.random.default_rng(1)
    out = []
    for a, others in PAIRS.items():
        for b in others:
            if a not in bc or b not in bc:
                continue
            items = sorted(set(bc[a]) & set(bc[b]))
            d = [np.mean(bc[a][i]) - np.mean(bc[b][i]) for i in items]
            boots = [float(np.mean([d[k] for k in rng.integers(len(d), size=len(d))])) for _ in range(BOOT)]
            out.append({"a": a, "b": b, "n": len(items), "diff": float(np.mean(d)),
                        "ci": [float(v) for v in np.percentile(boots, [2.5, 97.5])],
                        "p_one_sided": float(np.mean(np.array(boots) <= 0))})  # fmt: skip
    return out


def main():
    rows = [json.loads(line) for path in sys.argv[1:] for line in (Path(path) / "predictions.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r.get("correct") is not None]
    for s in stats(rows):
        lo, hi = s["ci"]
        print(f"{s['condition']:24s} n={s['n_cases']:4d} draws={s['n_draws']:5d}  %right {100 * s['accuracy']:.1f} "
              f"[{100 * lo:.1f}, {100 * hi:.1f}] (knew {100 * s['acc_knew']:.1f}, didn't {100 * s['acc_didnt']:.1f})")  # fmt: skip
    for x in paired(rows):
        lo, hi = x["ci"]
        print(f"{x['a']} − {x['b']}: {100 * x['diff']:+.1f}pp [{100 * lo:+.1f}, {100 * hi:+.1f}] p={x['p_one_sided']:.4f} (n={x['n']})")


if __name__ == "__main__":
    main()
