"""Score essay_grading_inline runs: how often does Qwen's own score match the investigator's?

    uv run python -m cfbench.evals.essay_grading_inline.score data/runs/cfbench/<date>_<slug> [...]

Per (version, condition): `share_bal` (headline) = `share` with each case weighted by 1 / (cases with its modal score),
so every modal score counts equally (Qwen's modal is 2 on 41% of cases; "always 2" would otherwise score well); `share`
= per draw, the share of Qwen's samples of that prompt whose score equals the submitted one, averaged over draws then
cases; `modal` = % of draws naming Qwen's most common score (cfbench.answer's measure); `mae` = mean |submitted - Qwen's
mean score|. 95% CIs bootstrap arms (an arm's 8 essays resampled together: cases on one arm are correlated). Then every
paired difference against olens_inline over the same cases, with its CI.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cfbench.core import canon_condition
from cfbench.evals.essay_grading_inline.eval import canon

BOOT = 2000
PAIRS = (("olens_inline", "olens_placebo_inline"), ("olens_inline", "transcript_inline"), ("olens_placebo_inline", "transcript_inline"))


def per_case(rows: list[dict]) -> dict[str, dict[str, dict[str, dict[tuple, float]]]]:
    """version -> metric -> condition -> {(arm, item): mean over draws}."""
    acc: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for r in rows:
        if not r.get("valid", True):
            continue
        a, dist = canon(r.get("answer")), r["answers_cf"]
        n = sum(dist.values())
        mean = sum(float(s) * k for s, k in dist.items()) / n
        key, cond = (r["cf_arm"], r["item"]), canon_condition(r["condition"])
        acc[r["version"]]["share"][cond][key].append(dist.get(a, 0) / n)
        acc[r["version"]]["modal"][cond][key].append(float(a == r["modal_cf"]))
        acc[r["version"]]["mae"][cond][key].append(abs(float(a) - mean) if a else 10.0)
    mean = lambda ks: {k: float(np.mean(x)) for k, x in ks.items()}
    return {v: {m: {c: mean(ks) for c, ks in cs.items()} for m, cs in ms.items()} for v, ms in acc.items()}


def boot(vals: dict[tuple, float], rng, w: dict[tuple, float] | None = None) -> tuple[float, float, float]:
    """(Weighted) mean over cases, 95% CI resampling arms."""
    by_arm = defaultdict(list)
    for k, x in vals.items():
        by_arm[k[0]].append((x * (w[k] if w else 1), w[k] if w else 1))
    arms = list(by_arm)
    sums, ns = np.array([sum(x for x, _ in by_arm[a]) for a in arms]), np.array([sum(v for _, v in by_arm[a]) for a in arms])
    idx = rng.integers(len(arms), size=(BOOT, len(arms)))
    means = sums[idx].sum(1) / ns[idx].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(sums.sum() / ns.sum()), float(lo), float(hi)


def main(paths: list[str]):
    rows = [json.loads(line) for p in paths for line in (Path(p) / "predictions.jsonl").open()]
    rng = np.random.default_rng(0)
    modal = {(r["version"], r["cf_arm"], r["item"]): r["modal_cf"] for r in rows}
    for v, ms in sorted(per_case(rows).items()):
        print(f"== {v}")
        count = Counter(m for (ver, *_), m in modal.items() if ver == v)
        w = {(a, i): 1 / count[m] for (ver, a, i), m in modal.items() if ver == v}
        ms = {"share_bal": ms["share"]} | ms
        for m, cs in ms.items():
            wm = w if m == "share_bal" else None
            for c, vals in sorted(cs.items()):
                mu, lo, hi = boot(vals, rng, wm)
                print(f"  {m:5} {c:22} {mu:7.3f} [{lo:.3f}, {hi:.3f}]  n={len(vals)}")
            for a, b in PAIRS:
                if a in cs and b in cs:
                    common = cs[a].keys() & cs[b].keys()
                    mu, lo, hi = boot({k: cs[a][k] - cs[b][k] for k in common}, rng, wm)
                    print(f"  {m:5} {a} - {b}: {mu:+.3f} [{lo:+.3f}, {hi:+.3f}]  n={len(common)}")


if __name__ == "__main__":
    main(sys.argv[1:])
