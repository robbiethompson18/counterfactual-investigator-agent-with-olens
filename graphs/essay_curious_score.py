"""Charts for the essay_curious out-of-10 prediction (docs/envs/essay_curious.md): RMSE of the predicted no-scenario mean
score per condition, with 95% case-bootstrap CIs and the "edit does nothing" null (predict the scenario arm's own mean),
paired differences, and a per-case plot of every draw against the truth. Rows come from scripts/honest_score_predict.py@d89747b4.

    uv run python -m graphs.essay_curious_score 'data/runs/cfbench/2026-10-28_essay-ten-*' [--out docs/img/essay-curious]
"""

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from graphs.style import GRID, INK, MUTED, OLENS, bar_kw, color, marker_kw

CONDS = ["no_transcript_score", "agentic_score", "olens_score", "olens_placebo_score"]
LABEL = {
    "no_transcript_score": "No transcript",
    "agentic_score": "Transcript-only",
    "olens_score": OLENS,
    "olens_placebo_score": f"{OLENS} placebo",
}
COLOR = {c: color(c) for c in CONDS}  # graphs.style: a placebo is its tool's colour, hatched (an open dot in the scatter)
N_BOOT = 4000


def load(pattern: str) -> tuple[list[tuple], dict, dict]:
    """Cases (item, base arm), condition -> case -> [predicted score per draw], case -> (base mean, cf mean)."""
    files = [Path(d, "predictions.jsonl") for d in glob.glob(pattern)]
    rows = [json.loads(line) for f in files if f.exists() for line in f.read_text().splitlines()]  # abandoned runs may have none
    by: dict = defaultdict(lambda: defaultdict(list))
    truth = {}
    for r in rows:
        if not r["valid"]:
            continue
        key = (r["env"].rsplit("@", 1)[1], r["base_arm"].split("/")[-1])
        by[r["condition"]][key].append(r["score"])
        truth[key] = (r["base_mean"], r["cf_mean"])
    return sorted(truth), by, truth


def case_mse(preds: list[float], cf: float) -> float:
    return float(np.mean([(p - cf) ** 2 for p in preds]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("--out", type=Path, default=Path("docs/img/essay-curious"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    keys, by, truth = load(a.pattern)
    conds = [c for c in CONDS if c in by]
    keys = [k for k in keys if all(k in by[c] for c in conds)]  # cases every condition has
    mse = {c: np.array([case_mse(by[c][k], truth[k][1]) for k in keys]) for c in conds}
    mse["null"] = np.array([(truth[k][0] - truth[k][1]) ** 2 for k in keys])
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(keys), len(keys)) for _ in range(N_BOOT)]
    rmse = lambda v: float(np.sqrt(v.mean()))
    stats = {}
    for c in [*conds, "null"]:
        boot = [rmse(mse[c][i]) for i in idx]
        stats[c] = (rmse(mse[c]), *np.percentile(boot, [2.5, 97.5]))
        name = LABEL.get(c, "edit does nothing")
        print(f"{name:17} RMSE {stats[c][0]:.2f}  [{stats[c][1]:.2f}, {stats[c][2]:.2f}]  ({len(keys)} cases)")
    for x, y in [("olens_score", "agentic_score"), ("olens_score", "olens_placebo_score"), ("agentic_score", "no_transcript_score")]:
        if x in mse and y in mse:
            d = [rmse(mse[x][i]) - rmse(mse[y][i]) for i in idx]
            print(
                f"{LABEL[x]} - {LABEL[y]}: {rmse(mse[x]) - rmse(mse[y]):+.2f} [{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}]"
            )

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, c in enumerate(conds):
        m, lo, hi = stats[c]
        ax.bar(i, m, width=0.65, **bar_kw(c))
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color=MUTED, capsize=4, lw=1)
        ax.text(i, hi + 0.05, f"{m:.2f}", ha="center", color=INK)
    ax.axhline(stats["null"][0], color=MUTED, ls="--", lw=1)
    ax.text(len(conds) - 0.5, stats["null"][0] + 0.04, f"edit does nothing {stats['null'][0]:.2f}", ha="right", color=MUTED, fontsize=9)
    ax.set_ylim(0, max(stats[c][2] for c in [*conds, "null"]) * 1.15)
    ax.set_xticks(range(len(conds)), [LABEL[c] for c in conds])
    ax.set_ylabel("RMSE of predicted mean score (points out of 10)")
    ax.set_title(
        f"Predict Qwen's mean score with the scenario removed ({len(keys)} cases, lower is better)", fontsize=10, color=INK, loc="left"
    )
    fig.tight_layout()
    fig.savefig(a.out / "score_rmse.png")

    order = sorted(keys, key=lambda k: (truth[k][1], k))
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(order) + 1.4))
    off = {c: (j - (len(conds) - 1) / 2) * 0.16 for j, c in enumerate(conds)}
    for i, k in enumerate(order):
        base, cf = truth[k]
        ax.plot([base, cf], [i, i], color=GRID, lw=3, zorder=1)
        ax.scatter([base], [i], marker="|", s=120, color=MUTED, zorder=2)
        ax.scatter([cf], [i], marker="D", s=40, color=INK, zorder=4)
        for c in conds:
            ax.scatter(by[c][k], [i + off[c]] * len(by[c][k]), s=14, alpha=0.75, zorder=3, **marker_kw(c))
    ax.set_yticks(range(len(order)), [f"{item}, {arm}" for item, arm in order], fontsize=8)
    ax.set_xlim(0, 10.3)
    ax.set_xlabel("score out of 10   (| scenario arm's mean,  ◆ no-scenario mean = truth,  dots = predictions)")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=COLOR[c], markerfacecolor=marker_kw(c)["facecolor"], label=LABEL[c]) for c in conds
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
    ax.set_title("Every prediction per case, sorted by the no-scenario mean")
    fig.tight_layout()
    fig.savefig(a.out / "score_per_case.png")


if __name__ == "__main__":
    main()
