"""mcq_unselected: Haiku predicting the letter Qwen gives on an unselected MMLU-Pro set, prompt only vs O-lens, J-lens and NLA
with their placebos, beside the linear probe and Qwen's own letter logits (the ceiling), for all cases, the cases where
Qwen's letter is the gold and the cases where it isn't. Bars are the mean over cases of the per-case accuracy over draws
(submitted letter equals Qwen's modal letter); whiskers are 95% bootstrap CIs over cases. A second figure shows the paired
differences between conditions, all cases and Qwen-wrong cases side by side. Reads the runs' predictions.jsonl, pooled over
both batches (docs/mcq-unselected.md).

    uv run python -m graphs.mcq_unselected [--runs data/runs/cfbench/<run> ...] [--out docs/img/mcq-unselected.png]
        [--out-diff docs/img/mcq-unselected-diff.png]
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["transcript_inline", "olens_inline", "olens_placebo_inline", "jlens_inline", "jlens_placebo_inline", "nla_inline",
          "nla_placebo_inline", "probe", "qwen_logits"]  # fmt: skip
LABEL = {"transcript_inline": "prompt only", "olens_inline": "O-lens", "olens_placebo_inline": "O-lens placebo",
         "jlens_inline": "J-lens", "jlens_placebo_inline": "J-lens placebo", "nla_inline": "NLA (1 draw)",
         "nla_placebo_inline": "NLA placebo (1 draw)", "probe": "linear probe", "qwen_logits": "Qwen's logits (ceiling)"}  # fmt: skip
# CLAUDE.md colours: text blue, O-lens orange, J++ purple; a placebo is its lens's colour with black stripes. NLA green as in
# cutoff_qa_inline; the probe magenta and the ceiling white with an ink outline, both outside the reserved colours.
COLOR = {"transcript_inline": "#2a78d6", "olens_inline": "#eb6834", "olens_placebo_inline": "#eb6834",
         "jlens_inline": "#8e44ad", "jlens_placebo_inline": "#8e44ad", "nla_inline": "#2e9e5b", "nla_placebo_inline": "#2e9e5b",
         "probe": "#c2185b", "qwen_logits": "white"}  # fmt: skip
HATCH = {"olens_placebo_inline": "////", "jlens_placebo_inline": "////", "nla_placebo_inline": "////"}
RUNS = [Path(f"data/runs/cfbench/2026-10-29_mcq-unselected-haiku-{s}") for s in ("b1r", "b2", "jlens-b1", "jlens-b2", "nla-b1", "nla-b2")]
RUNS.append(Path("data/runs/mcq_answer_probe/2026-10-29_mcq-unselected-probe"))
GROUPS = {"all cases": None, "Qwen's letter is the gold": True, "Qwen's letter is wrong": False}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True}
)


def per_case(runs: list[Path]) -> dict[str, dict[str, tuple[bool, float]]]:
    """condition -> case -> (Qwen's modal letter is the gold, mean accuracy over draws), pooled over the runs."""
    acc: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    right = {}
    for line in (ln for run in runs for ln in (run / "predictions.jsonl").read_text().splitlines()):
        r = json.loads(line)
        acc[r["condition"]][r["item"]].append(bool(r["correct"]))
        right[r["item"]] = r["modal_cf"] == r["gold"]
    return {c: {i: (right[i], sum(v) / len(v)) for i, v in cases.items()} for c, cases in acc.items()}


def ci(xs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def edge(s: str) -> str | None:
    return INK if s in HATCH or s == "qwen_logits" else None


# the difference chart: (question, [(a, b), ...]) groups, each row a − b
PAIRS = [
    (
        "Lens vs prompt only",
        [("olens_inline", "transcript_inline"), ("jlens_inline", "transcript_inline"), ("nla_inline", "transcript_inline")],
    ),
    (
        "Does the lens beat its placebo?",
        [("olens_inline", "olens_placebo_inline"), ("jlens_inline", "jlens_placebo_inline"), ("nla_inline", "nla_placebo_inline")],
    ),
    (
        "Does the placebo alone help?",
        [
            ("olens_placebo_inline", "transcript_inline"),
            ("jlens_placebo_inline", "transcript_inline"),
            ("nla_placebo_inline", "transcript_inline"),
        ],
    ),
    ("Between lenses", [("jlens_inline", "olens_inline"), ("olens_inline", "nla_inline"), ("jlens_inline", "nla_inline")]),
    ("What a probe reads from the same activations", [("probe", "transcript_inline"), ("probe", "jlens_inline")]),
]


def diffs(runs: list[Path], out: Path) -> None:
    """A row per pair, grouped by the question it answers, one panel for all cases and one for the Qwen-wrong cases: the
    mean paired difference in per-case accuracy (pp) with its 95% bootstrap CI over cases, filled when the CI excludes 0."""
    data = per_case(runs)
    rows: list[tuple[str, tuple[str, str] | None]] = [r for q, pairs in PAIRS for r in [(q, None), *((q, p) for p in pairs)]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.4 + 0.34 * len(rows)), sharey=True, gridspec_kw={"wspace": 0.08})
    labels = [q if pair is None else f"{LABEL[pair[0]]} − {LABEL[pair[1]]}" for q, pair in rows]
    for ax, (title, right) in zip(axes, [("all cases", None), ("Qwen's letter is wrong", False)], strict=True):
        for k, (q, pair) in enumerate(rows):
            if pair is None:  # a group header
                continue
            y = len(rows) - 1 - k
            a, b = pair
            ds = [100 * (data[a][i][1] - data[b][i][1]) for i in data[a] if right is None or data[a][i][0] == right]
            m, (lo, hi) = sum(ds) / len(ds), ci(ds)
            ax.plot([lo, hi], [y, y], color=INK, lw=1.6, solid_capstyle="butt", zorder=3)
            ax.plot(m, y, "o", ms=6, zorder=4, mec=INK, mfc=INK if lo > 0 or hi < 0 else "white")
            ax.text(hi + 1.5, y, f"{m:+.1f}  [{lo:+.1f}, {hi:+.1f}]", va="center", fontsize=8, color=INK)
        ax.axvline(0, color=MUTED, lw=0.8, zorder=2)
        ax.set_yticks(range(len(rows)))
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(-15, 95)
        ax.grid(axis="y", visible=False)
        n = sum(1 for r, _ in data["transcript_inline"].values() if right is None or r == right)
        ax.set_title(f"{title} (n = {n})", loc="left", fontsize=10, color=INK)
    axes[0].set_yticklabels(reversed(labels), fontsize=9)
    for tick, (_, pair) in zip(reversed(axes[0].get_yticklabels()), rows, strict=True):
        if pair is None:
            tick.set_fontweight("bold")
            tick.set_color(MUTED)
    axes[1].set_xlabel("difference in % of cases right, paired by case (95% bootstrap CI; filled = excludes 0)", fontsize=8.5, loc="right")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(out)


def main(runs: list[Path], out: Path) -> None:
    data = per_case(runs)
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    width, gap = 0.095, 0.006
    for i, (group, right) in enumerate(GROUPS.items()):
        for j, s in enumerate(SERIES):
            xs = [a for r, a in data[s].values() if right is None or r == right]
            m, (lo, hi) = sum(xs) / len(xs), ci(xs)
            x = i + (j - (len(SERIES) - 1) / 2) * (width + gap)
            ax.bar(x, 100 * m, width, color=COLOR[s], hatch=HATCH.get(s), edgecolor=edge(s), lw=0.8 if s == "qwen_logits" else 0, zorder=3)
            yerr = [[100 * (m - lo)], [100 * (hi - m)]]
            ax.errorbar(x, 100 * m, yerr=yerr, fmt="none", ecolor=MUTED, elinewidth=1, capsize=2.5, zorder=4)
            ax.text(x, 100 * hi + 1.5, f"{100 * m:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK)
        n = sum(1 for r, _ in data[SERIES[0]].values() if right is None or r == right)
        ax.text(i, -0.15, f"n = {n}", ha="center", va="top", fontsize=8, color=MUTED, transform=ax.get_xaxis_transform())
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels(GROUPS.keys(), fontsize=9)
    ax.set_xlim(-0.55, len(GROUPS) - 0.45)
    ax.set_ylim(0, 110)
    ax.set_ylabel("% of cases where the letter Qwen gives is named")
    ax.grid(axis="x", visible=False)
    ax.set_title("Predicting Qwen3.6-27B's MMLU-Pro letter from the bare prompt (Haiku 5.5, 3 draws; unselected set, both batches)",
                 loc="left", fontsize=10.5, color=INK, pad=38)  # fmt: skip
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR[s], hatch=HATCH.get(s), edgecolor=edge(s), lw=0.8 if s == "qwen_logits" else 0)
               for s in SERIES]  # fmt: skip
    ax.legend(handles, [LABEL[s] for s in SERIES], loc="lower left", frameon=False, fontsize=8, ncol=5, bbox_to_anchor=(0, 1.0))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", default=RUNS)
    p.add_argument("--out", type=Path, default=Path("docs/img/mcq-unselected.png"))
    p.add_argument("--out-diff", type=Path, default=Path("docs/img/mcq-unselected-diff.png"))
    a = p.parse_args()
    main(a.runs, a.out)
    diffs(a.runs, a.out_diff)
