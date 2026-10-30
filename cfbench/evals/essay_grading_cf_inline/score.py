"""Score essay_grading_cf_inline runs: P(removing the opener raises Qwen's score).

    uv run python -m cfbench.evals.essay_grading_cf_inline.score data/runs/cfbench/<date>_<slug> [...]

Per condition: % right (guess = p >= 0.5, per case the mean over draws), AUROC of the mean p (up vs not), Brier, each with
a 95% CI resampling essays (truths are per essay); the same on the clear cases (|shift| >= 0.5 points); then every paired
difference against olens_inline over the same cases.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from cfbench.core import canon_condition

BOOT = 1000
PAIRS = (("olens_inline", "olens_placebo_inline"), ("olens_inline", "transcript_inline"), ("olens_placebo_inline", "transcript_inline"))


def auroc(p: np.ndarray, y: np.ndarray) -> float:
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def metrics(ps: dict, truth: dict, keys: list) -> dict[str, float]:
    p, y = np.array([ps[k] for k in keys]), np.array([truth[k] for k in keys])
    return {"right": float(((p >= 0.5) == y).mean()), "brier": float(((p - y) ** 2).mean()),
            "auroc": auroc(p, y)}  # fmt: skip


def boot(f, keys: list, rng) -> list[dict]:
    """f(keys) over BOOT resamples of essays (all of an essay's cases together)."""
    by = defaultdict(list)
    for k in keys:
        by[k[1]].append(k)
    items = list(by)
    return [f([k for i in rng.integers(len(items), size=len(items)) for k in by[items[i]]]) for _ in range(BOOT)]


def main(paths: list[str]):
    rows = [json.loads(line) for p in paths for line in (Path(p) / "predictions.jsonl").open()]
    ps, truth, shift = defaultdict(lambda: defaultdict(list)), {}, {}
    for r in rows:
        if r.get("valid", True) and r.get("p") is not None:
            k = (r["base_arm"], r["item"])
            ps[canon_condition(r["condition"])][k].append(r["p"])
            truth[k], shift[k] = r["k_cf"], r["shift"]
    mp = {c: {k: float(np.mean(v)) for k, v in d.items()} for c, d in ps.items()}
    rng = np.random.default_rng(0)
    for name, keep in (("all", lambda k: True), ("clear |shift|>=0.5", lambda k: abs(shift[k]) >= 0.5)):
        print(f"== {name}")
        for c, d in sorted(mp.items()):
            keys = [k for k in d if keep(k)]
            m = metrics(d, truth, keys)
            bs = boot(lambda ks, d=d: metrics(d, truth, ks), keys, rng)
            ci = {s: np.nanpercentile([b[s] for b in bs], [2.5, 97.5]) for s in m}
            up = np.mean([truth[k] for k in keys])
            print(f"  {c:22} n={len(keys)} up={up:.2f}  " + "  ".join(f"{s} {m[s]:.3f} [{ci[s][0]:.3f}, {ci[s][1]:.3f}]" for s in m))
        for a, b in PAIRS:
            if a in mp and b in mp:
                keys = [k for k in mp[a].keys() & mp[b].keys() if keep(k)]
                diff = lambda ks, a=a, b=b: {
                    s: metrics(mp[a], truth, ks)[s] - metrics(mp[b], truth, ks)[s] for s in ("right", "auroc", "brier")
                }
                m, bs = diff(keys), boot(diff, keys, rng)
                print(f"  {a} - {b}: " + "  ".join(f"{s} {m[s]:+.3f} [{np.nanpercentile([x[s] for x in bs], 2.5):+.3f}, "
                                                   f"{np.nanpercentile([x[s] for x in bs], 97.5):+.3f}]" for s in m))  # fmt: skip


if __name__ == "__main__":
    main(sys.argv[1:])
