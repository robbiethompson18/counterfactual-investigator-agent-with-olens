"""Score hint_qwen_dsv4 runs (the `knew` variant): does the investigator guess right whether Qwen answers correctly
without the hint? Headline = % right, guess = p >= 0.5 (the framework's claim task collects a probability; LLM
probabilities are poorly calibrated, so the Brier score is not the headline). Classes are balanced, so % right is the
balanced accuracy and chance is 50%.

    uv run python -m cfbench.evals.hint_qwen_dsv4.score data/runs/cfbench/<date>_<slug> [...]

Per (condition, version), never pooled over versions: % right with a 95% bootstrap CI over items (an item's draws
resampled together), % right per class, AUROC of p (knew vs didn't; calibration-free), mean Brier for reference; then the
paired comparisons of O-lens against placebo, transcript tools and no transcript: per item the mean of its draws' right/wrong,
the mean difference over items with a bootstrap CI, and a sign test over the items that differ. `stats(rows)` is what the env
viewer's claim table reads (environments/_lib/evals.py, the eval's `score` module); `accuracy` is its headline column.
"""

import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

from cfbench.core import canon_condition
from cfbench.evals.hint_qwen_dsv4.eval import DIDNT, KNEW
from environments._lib.evals import auroc

CLASSES = (KNEW, DIDNT)
BOOT = 2000
OLENS = "olens_agentic"
VERSUS = ("olens_placebo_agentic", "transcript_agentic", "no_transcript")


def boot_mean(per_item: list[float], rng: np.random.Generator) -> list[float]:
    """95% bootstrap CI of the mean over items."""
    x = np.array(per_item)
    return [float(v) for v in np.percentile([x[rng.integers(len(x), size=len(x))].mean() for _ in range(BOOT)], [2.5, 97.5])]


def sign_test(diffs: list[float]) -> tuple[int, int, float]:
    pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    n = pos + neg
    return pos, neg, (min(1.0, 2 * sum(comb(n, k) for k in range(max(pos, neg), n + 1)) / 2**n) if n else 1.0)


def paired(right: dict[tuple[str, str], dict[str, list[bool]]], rng: np.random.Generator) -> list[dict]:
    """O-lens vs each other condition per version: mean over shared items of (O-lens − other) per-item % right."""
    out = []
    for version in sorted({v for _, v in right}):
        a = right.get((OLENS, version), {})
        for other in VERSUS:
            b = right.get((other, version), {})
            items = sorted(set(a) & set(b))
            if not items:
                continue
            d = [float(np.mean(a[i]) - np.mean(b[i])) for i in items]
            pos, neg, p = sign_test(d)
            out.append({"version": version, "a": OLENS, "b": other, "n": len(items), "diff": float(np.mean(d)),
                        "ci": boot_mean(d, rng), "better": pos, "worse": neg, "p_sign": p})  # fmt: skip
    return out


def stats(rows: list[dict]) -> list[dict]:
    """One row per (condition, version): accuracy (headline) with CI, per-class accuracy, AUROC, mean Brier, invalid."""
    invalid = defaultdict(int)
    by = defaultdict(lambda: defaultdict(list))  # (condition, version) -> item -> [(p, y)]
    cls = {}
    for r in rows:
        key = (canon_condition(r["condition"]), r["version"])
        if not r["valid"]:
            invalid[key] += 1
            continue
        by[key][r["item"]].append((r["p"], r["k_cf"] // r["n_cf"]))
        cls[r["version"], r["item"]] = r["model"]
    rng, out = np.random.default_rng(0), []
    for (cond, version), items in sorted(by.items()):
        right = {i: [float((p >= 0.5) == bool(y)) for p, y in xs] for i, xs in items.items()}
        per_item = [float(np.mean(v)) for v in right.values()]
        ps = {m: [p for i, xs in items.items() if cls[version, i] == m for p, _ in xs] for m in CLASSES}
        out.append({
            "condition": cond, "version": version, "invalid": invalid[cond, version], "n": sum(len(xs) for xs in items.values()),
            "n_items": len(items), "accuracy": float(np.mean(per_item)), "accuracy_ci": boot_mean(per_item, rng),
            "accuracy_by_class": {m: float(np.mean([np.mean(right[i]) for i in items if cls[version, i] == m])) for m in CLASSES if ps[m]},
            "pbar": {m: float(np.mean(ps[m])) for m in CLASSES if ps[m]}, "auroc": auroc(ps[KNEW], ps[DIDNT]),
            "mean_brier": float(np.mean([(p - y) ** 2 for xs in items.values() for p, y in xs])),
        })  # fmt: skip
    return out


def right_by_item(rows: list[dict], correct) -> dict[tuple[str, str], dict[str, list[bool]]]:
    out: dict[tuple[str, str], dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["valid"]:
            out[canon_condition(r["condition"]), r["version"]][r["item"]].append(bool(correct(r)))
    return out


def main():
    rows = [json.loads(line) for path in sys.argv[1:] for line in (Path(path) / "predictions.jsonl").read_text().splitlines()]
    print("guess = p >= 0.5; chance 50% (balanced classes)")
    for s in stats(rows):
        lo, hi = s["accuracy_ci"]
        by = " / ".join(f"{s['accuracy_by_class'].get(m, float('nan')):.0%}" for m in CLASSES)
        print(f"{s['condition'] + ' ' + s['version']:36} items {s['n_items']:>3}  right {s['accuracy']:.0%} [{lo:.0%}, {hi:.0%}]  "
              f"knew/didn't {by}  AUROC {s['auroc'] or float('nan'):.2f}  Brier {s['mean_brier']:.3f}  invalid {s['invalid']}")  # fmt: skip
    for x in paired(right_by_item(rows, lambda r: (r["p"] >= 0.5) == bool(r["k_cf"] // r["n_cf"])), np.random.default_rng(1)):
        lo, hi = x["ci"]
        print(f"{x['version']:11} {x['a']} − {x['b']}: {x['diff']:+.0%} [{lo:+.0%}, {hi:+.0%}] over {x['n']} items; "
              f"sign {x['better']}/{x['worse']} p = {x['p_sign']:.3g}")  # fmt: skip


if __name__ == "__main__":
    main()
