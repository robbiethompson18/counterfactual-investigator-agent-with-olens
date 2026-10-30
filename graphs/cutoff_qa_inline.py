"""cutoff_qa_inline: Haiku predicting Qwen's answer from the question, transcript-only vs O-lens, J++ and NLA, each with its
placebo, one group of seven bars for all items, for the stale items (gate_class stable_wrong: Qwen's answer is out of date)
and for the rest (stable_right). Bars are the mean over items of the per-item accuracy over draws (exact match to Qwen's
modal answer); whiskers are 95% bootstrap CIs over items. A second, small figure shows the paired differences between
conditions (per item, then the mean) with their 95% bootstrap CIs, all items. Reads the runs' predictions.jsonl, pooled
(one run per condition on the 999-item inputs, cfbench_evals/2026-10-29_cutoff-qa-inline-inputs-1k; docs/cutoff-qa-inline.md).

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

from graphs.style import INK, MUTED, bar_kw, diff_rows, use
from graphs.style import color as _color
from graphs.style import label as _label

use()
SERIES = ["transcript_inline", "olens_inline", "olens_placebo_inline", "jlens_inline", "jlens_placebo_inline", "nla_inline",
          "nla_placebo_inline"]  # fmt: skip
LABEL = {s: _label(s) for s in SERIES}  # graphs.style's, kept here for the modules that import them
COLOR = {s: _color(s) for s in SERIES}
HATCH = {s: bar_kw(s)["hatch"] for s in SERIES if bar_kw(s)["hatch"]}
SLUGS = ("transcript", "olens", "olens-placebo", "jlens", "jlens-placebo", "nla", "nla-placebo-mmlu")
RUNS = [Path(f"data/runs/cfbench/2026-10-29_cutoff-qa-1k-{s}") for s in SLUGS]  # one run per condition, 999 items
GROUPS = {
    "all items": None,
    "stale (Qwen's answer\nis out of date)": "stable_wrong",
    "not stale (Qwen's\nanswer still true)": "stable_right",
}


def per_item(runs: list[Path], by: str = "gate_class") -> dict[str, dict[str, tuple[str, float]]]:
    """condition -> item -> (the row's `by` field, mean accuracy over draws), pooled over the runs."""
    acc: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    cls = {}
    for line in (ln for run in runs for ln in (run / "predictions.jsonl").read_text().splitlines()):
        r = json.loads(line)
        acc[r["condition"]][r["item"]].append(bool(r["correct"]))
        cls[r["item"]] = r[by]
    return {c: {i: (cls[i], sum(v) / len(v)) for i, v in items.items()} for c, items in acc.items()}


def ci(xs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


# the difference chart: (question, [(a, b), ...]) groups, each row a − b
PAIRS = [
    ("Does the lens beat its placebo?", [(f"{lens}_inline", f"{lens}_placebo_inline") for lens in ("olens", "jlens", "nla")]),
    ("Does the placebo alone help?", [(f"{lens}_placebo_inline", "transcript_inline") for lens in ("olens", "jlens", "nla")]),
    ("Lens vs transcript only", [(f"{lens}_inline", "transcript_inline") for lens in ("olens", "jlens", "nla")]),
    ("Between lenses", [("olens_inline", "jlens_inline"), ("olens_inline", "nla_inline"), ("jlens_inline", "nla_inline")]),
]


def diffs(runs: list[Path], out: Path, xlim: tuple[float, float] = (-8, 16)) -> None:
    """A row per pair, grouped by the question it answers: the mean paired difference in per-item accuracy (pp) with its 95%
    bootstrap CI over items (graphs.style.diff_rows: filled when the CI excludes 0, labelled by the two conditions' swatches)."""
    data = per_item(runs)
    rows: list = []
    for q, pairs in PAIRS:
        rows.append(q)
        for a, b in pairs:
            ds = [100 * (data[a][i][1] - data[b][i][1]) for i in data[a]]
            rows.append((a, b, sum(ds) / len(ds), *ci(ds)))
    fig, ax = plt.subplots(figsize=(6.2, 0.35 + 0.34 * len(rows)))
    ax.set_xlim(*xlim)
    diff_rows(ax, rows)
    ax.set_xlabel("difference in % of items right, paired by item (95% bootstrap CI; filled = excludes 0)", fontsize=8.5)
    ax.set_title(f"All {len(data['transcript_inline'])} items", loc="left", fontsize=10, color=INK)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(out)


def bar_chart(runs: list[Path], out: Path, by: str, groups: dict, title: str, figsize: tuple[float, float] = (10.5, 4.1)) -> None:
    """Seven bars (SERIES) per group: the mean over items of per-item accuracy over draws, 95% bootstrap CIs over items. `groups`
    maps a group's label to a value of the rows' `by` field (None: every item)."""
    data = per_item(runs, by=by)
    fig, ax = plt.subplots(figsize=figsize)
    width, gap = 0.115, 0.008
    for i, value in enumerate(groups.values()):
        for j, s in enumerate(SERIES):
            xs = [a for c, a in data[s].values() if value is None or c == value]
            m, (lo, hi) = sum(xs) / len(xs), ci(xs)
            x = i + (j - (len(SERIES) - 1) / 2) * (width + gap)
            ax.bar(x, 100 * m, width, zorder=3, **bar_kw(s))
            ax.errorbar(x, 100 * m, yerr=[[100 * (m - lo)], [100 * (hi - m)]], fmt="none", ecolor=MUTED, elinewidth=1, capsize=3, zorder=4)
            ax.text(x, 100 * hi + 1.5, f"{100 * m:.0f}", ha="center", va="bottom", fontsize=8 if len(groups) < 5 else 7.5, color=INK)
        n = sum(1 for c, _ in data[SERIES[0]].values() if value is None or c == value)
        ax.text(i, -0.15, f"n = {n}", ha="center", va="top", fontsize=8, color=MUTED, transform=ax.get_xaxis_transform())
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups.keys(), fontsize=9)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.set_ylim(0, 108)
    ax.set_ylabel("% of items where Haiku names Qwen's answer")
    ax.grid(axis="x", visible=False)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=24)
    handles = [plt.Rectangle((0, 0), 1, 1, **bar_kw(s)) for s in SERIES]
    ax.legend(handles, [LABEL[s] for s in SERIES], loc="lower left", frameon=False, fontsize=8.5, ncol=7, bbox_to_anchor=(0, 1.0))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(out)


def main(runs: list[Path], out: Path) -> None:
    bar_chart(runs, out, "gate_class", GROUPS, "Predicting Qwen3.6-27B's answer to a factual question from the prompt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", default=RUNS)
    p.add_argument("--out", type=Path, default=Path("docs/img/cutoff-qa-inline.png"))
    p.add_argument("--out-diff", type=Path, default=Path("docs/img/cutoff-qa-inline-diff.png"))
    a = p.parse_args()
    main(a.runs, a.out)
    diffs(a.runs, a.out_diff)
