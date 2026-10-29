"""cutoff_qa_inline: Haiku predicting Qwen's answer from the question, transcript-only vs O-lens vs its placebo vs J++ vs its
placebo, one group of five bars for all items, for the stale items (gate_class stable_wrong: Qwen's answer is out of date)
and for the rest (stable_right). Bars are the mean over items of the per-item accuracy over draws (exact match to Qwen's
modal answer); whiskers are 95% bootstrap CIs over items. A second, small figure shows the paired differences between
conditions (per item, then the mean) with their 95% bootstrap CIs, all items. Reads the runs' predictions.jsonl, pooled
(the O-lens run has transcript_inline and the O-lens conditions, the J++ runs one condition each; docs/cutoff-qa-inline.md,
PR #393 for J++).

    uv run python -m graphs.cutoff_qa_inline [--runs data/runs/cfbench/<run> ...] [--out docs/img/cutoff-qa-inline.png]
        [--out-diff docs/img/cutoff-qa-inline-diff.png]
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["transcript_inline", "olens_inline", "olens_placebo_inline", "jlens_inline", "jlens_placebo_inline"]
LABEL = {"transcript_inline": "transcript only", "olens_inline": "O-lens", "olens_placebo_inline": "O-lens placebo",
         "jlens_inline": "J++", "jlens_placebo_inline": "J++ placebo"}  # fmt: skip
# J++ is purple (outside CLAUDE.md's reserved colours); a placebo is its lens's colour with black stripes (CLAUDE.md)
COLOR = {"transcript_inline": "#2a78d6", "olens_inline": "#eb6834", "olens_placebo_inline": "#eb6834",
         "jlens_inline": "#8e44ad", "jlens_placebo_inline": "#8e44ad"}  # fmt: skip
HATCH = {"olens_placebo_inline": "////", "jlens_placebo_inline": "////"}
RUNS = [Path(f"data/runs/cfbench/2026-10-29_cutoff-qa-haiku-200{s}") for s in ("", "-jlens", "-jlens-placebo")]
GROUPS = {
    "all items": None,
    "stale (Qwen's answer\nis out of date)": "stable_wrong",
    "not stale (Qwen's\nanswer still true)": "stable_right",
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True}
)


def per_item(runs: list[Path]) -> dict[str, dict[str, tuple[str, float]]]:
    """condition -> item -> (gate_class, mean accuracy over draws), pooled over the runs."""
    acc: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    cls = {}
    for line in (ln for run in runs for ln in (run / "predictions.jsonl").read_text().splitlines()):
        r = json.loads(line)
        acc[r["condition"]][r["item"]].append(bool(r["correct"]))
        cls[r["item"]] = r["gate_class"]
    return {c: {i: (cls[i], sum(v) / len(v)) for i, v in items.items()} for c, items in acc.items()}


def ci(xs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


# the difference chart: (question, [(a, b), ...]) groups, each row a − b
PAIRS = [
    ("Does the lens beat its placebo?", [("olens_inline", "olens_placebo_inline"), ("jlens_inline", "jlens_placebo_inline")]),
    ("Does the placebo alone help?", [("olens_placebo_inline", "transcript_inline"), ("jlens_placebo_inline", "transcript_inline")]),
    ("Lens vs transcript only", [("olens_inline", "transcript_inline"), ("jlens_inline", "transcript_inline")]),
    ("O-lens vs J++", [("olens_inline", "jlens_inline")]),
]


def diffs(runs: list[Path], out: Path) -> None:
    """A row per pair, grouped by the question it answers: the mean paired difference in per-item accuracy (pp) with its 95%
    bootstrap CI over items, filled when the CI excludes 0. One ink for every row: the label says which conditions."""
    data = per_item(runs)
    rows: list[tuple[str, tuple[str, str] | None]] = [r for q, pairs in PAIRS for r in [(q, None), *((q, p) for p in pairs)]]
    fig, ax = plt.subplots(figsize=(6.2, 0.35 + 0.34 * len(rows)))
    labels = []
    for k, (q, pair) in enumerate(rows):
        y = len(rows) - 1 - k
        if pair is None:  # a group header
            labels.append(q)
            continue
        a, b = pair
        ds = [100 * (data[a][i][1] - data[b][i][1]) for i in data[a]]
        m, (lo, hi) = sum(ds) / len(ds), ci(ds)
        ax.plot([lo, hi], [y, y], color=INK, lw=1.6, solid_capstyle="butt", zorder=3)
        ax.plot(m, y, "o", ms=6, zorder=4, mec=INK, mfc=INK if lo > 0 or hi < 0 else "white")
        ax.text(hi + 1, y, f"{m:+.1f}  [{lo:+.1f}, {hi:+.1f}]", va="center", fontsize=8, color=INK)
        labels.append(f"{LABEL[a]} − {LABEL[b]}")
    ax.axvline(0, color=MUTED, lw=0.8, zorder=2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(reversed(labels), fontsize=9)
    for tick, (_, pair) in zip(reversed(ax.get_yticklabels()), rows, strict=True):
        if pair is None:
            tick.set_fontweight("bold")
            tick.set_color(MUTED)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("difference in % of items right, paired by item (95% bootstrap CI; filled = excludes 0)", fontsize=8.5)
    ax.set_xlim(-12, 30)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"All {len(data['transcript_inline'])} items", loc="left", fontsize=10, color=INK)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(out)


def main(runs: list[Path], out: Path) -> None:
    data = per_item(runs)
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    width, gap = 0.16, 0.01
    for i, (group, gate) in enumerate(GROUPS.items()):
        for j, s in enumerate(SERIES):
            xs = [a for c, a in data[s].values() if gate is None or c == gate]
            m, (lo, hi) = sum(xs) / len(xs), ci(xs)
            x = i + (j - (len(SERIES) - 1) / 2) * (width + gap)
            ax.bar(x, 100 * m, width, color=COLOR[s], hatch=HATCH.get(s), edgecolor=INK if s in HATCH else None,
                   lw=0, zorder=3)  # fmt: skip
            ax.errorbar(x, 100 * m, yerr=[[100 * (m - lo)], [100 * (hi - m)]], fmt="none", ecolor=MUTED, elinewidth=1, capsize=3, zorder=4)
            ax.text(x, 100 * hi + 1.5, f"{100 * m:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
        n = sum(1 for c, _ in data[SERIES[0]].values() if gate is None or c == gate)
        ax.text(i, -0.15, f"n = {n}", ha="center", va="top", fontsize=8, color=MUTED, transform=ax.get_xaxis_transform())
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels(GROUPS.keys(), fontsize=9)
    ax.set_xlim(-0.55, len(GROUPS) - 0.45)
    ax.set_ylim(0, 108)
    ax.set_ylabel("% of items where Haiku names Qwen's answer")
    ax.grid(axis="x", visible=False)
    ax.set_title("Predicting Qwen3.6-27B's answer to a factual question from the prompt", loc="left", fontsize=10.5, color=INK, pad=24)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR[s], hatch=HATCH.get(s), edgecolor=INK if s in HATCH else None,
                             lw=0) for s in SERIES]  # fmt: skip
    ax.legend(handles, [LABEL[s] for s in SERIES], loc="lower left", frameon=False, fontsize=8.5, ncol=5, bbox_to_anchor=(0, 1.0))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", default=RUNS)
    p.add_argument("--out", type=Path, default=Path("docs/img/cutoff-qa-inline.png"))
    p.add_argument("--out-diff", type=Path, default=Path("docs/img/cutoff-qa-inline-diff.png"))
    a = p.parse_args()
    main(a.runs, a.out)
    diffs(a.runs, a.out_diff)
