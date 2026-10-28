"""Score hint_qwen_dsv4 runs: Brier of P(Qwen answers correctly without the hint), truth 1 for the items it knew, 0 for
the others; af_which's table (cfbench/evals/af_which/score.py) with the two classes in place of its two models.

    uv run python -m cfbench.evals.hint_qwen_dsv4.score data/runs/cfbench/<date>_<slug> [...]

Per (condition, version), never pooled over versions: mean Brier over predictions, Brier of each class's mean p (averaged
over the two classes), per-class mean Brier and mean p, the AUROC of p (knew vs didn't), accuracy at 0.5, and 95%
bootstrap CIs of both Briers over items. Always answering 0.5 scores 0.25 on both. `stats(rows)` is what the env viewer's
claim table reads (environments/_lib/evals.py, the eval's `score` module).
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cfbench.core import canon_condition
from cfbench.evals.hint_qwen_dsv4.eval import DIDNT, KNEW
from environments._lib.evals import auroc

CLASSES = (KNEW, DIDNT)
BOOT = 2000

Preds = list[tuple[str, str, float, int]]  # (item key, class, p, y)


def briers(preds: Preds) -> tuple[float, float]:
    pbar = {m: np.mean([p for _, mm, p, _ in preds if mm == m]) for m in CLASSES if any(mm == m for _, mm, _, _ in preds)}
    return float(np.mean([(p - y) ** 2 for _, _, p, y in preds])), float(np.mean([(pbar[m] - (m == KNEW)) ** 2 for m in pbar]))


def ci(preds: Preds, rng: np.random.Generator) -> np.ndarray:
    by_item = defaultdict(list)
    for x in preds:
        by_item[x[0]].append(x)
    items = list(by_item.values())
    stats_ = [briers([x for i in rng.integers(len(items), size=len(items)) for x in items[i]]) for _ in range(BOOT)]
    return np.percentile(stats_, [2.5, 97.5], axis=0)


def stat(preds: Preds, rng: np.random.Generator) -> dict:
    mean, of_mean = briers(preds)
    (lo_m, lo_b), (hi_m, hi_b) = ci(preds, rng)
    ps = {m: [p for _, mm, p, _ in preds if mm == m] for m in CLASSES}
    return {
        "n": len(preds),
        "mean_brier": mean,
        "mean_brier_ci": [float(lo_m), float(hi_m)],
        "brier_pbar": of_mean,
        "brier_pbar_ci": [float(lo_b), float(hi_b)],
        "model_brier": {m: float(np.mean([(p - (m == KNEW)) ** 2 for p in ps[m]])) for m in CLASSES if ps[m]},
        "pbar": {m: float(np.mean(ps[m])) for m in CLASSES if ps[m]},
        "auroc": auroc(ps[KNEW], ps[DIDNT]),
        "accuracy": float(np.mean([(p >= 0.5) == bool(y) for _, _, p, y in preds])),
    }


def stats(rows: list[dict]) -> list[dict]:
    """One row per (condition, version), invalid rows counted per (condition, version)."""
    invalid = Counter((canon_condition(r["condition"]), r["version"]) for r in rows if not r["valid"])
    preds: dict[tuple[str, str], Preds] = defaultdict(list)
    for r in rows:
        if r["valid"]:
            item, y = f"{r['version']}/{r['item']}", r["k_cf"] // r["n_cf"]
            preds[canon_condition(r["condition"]), r["version"]].append((item, r["model"], r["p"], y))
    rng, out = np.random.default_rng(0), []
    for cond, version in sorted(preds):  # versions are never pooled: no "pooled" row
        out.append({"condition": cond, "version": version, "invalid": invalid[cond, version]} | stat(preds[cond, version], rng))
    return out


def main():
    rows = [json.loads(line) for path in sys.argv[1:] for line in (Path(path) / "predictions.jsonl").read_text().splitlines()]
    print("always 0.5 scores 0.25 on both Briers")
    head = f"{'condition / version':36} {'n':>4}  {'mean Brier [95% CI]':22}  {'Brier(p_bar) [95% CI]':22}"
    print(f"{head}  {'p_bar knew / didnt':19}  AUROC  acc@0.5")
    for s in stats(rows):
        (lo_m, hi_m), (lo_b, hi_b) = s["mean_brier_ci"], s["brier_pbar_ci"]
        pb = " / ".join(f"{s['pbar'].get(m, float('nan')):.2f}" for m in CLASSES)
        print(
            f"{s['condition'] + ' ' + s['version']:36} {s['n']:>4}  {s['mean_brier']:.3f} [{lo_m:.3f},{hi_m:.3f}]  "
            f"{s['brier_pbar']:.3f} [{lo_b:.3f},{hi_b:.3f}]  {pb:19}  {s['auroc'] or float('nan'):.3f}  {s['accuracy']:.2f}"
        )


if __name__ == "__main__":
    main()
