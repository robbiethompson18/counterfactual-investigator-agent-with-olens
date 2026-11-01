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
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt

from cfbench.catalog import CUTOFF_QA_INLINE_RUNS
from cfbench.scoring import grouped_mean_ci
from graphs.style import MUTED, ci_bar, diff_rows, positions

# the bars, each lens beside its placebo under one label
BARS = [("transcript", ["transcript_inline"]), ("O-lens", ["olens_inline", "olens_placebo_inline"]),
        ("J++", ["jlens_inline", "jlens_placebo_inline"]), ("NLA", ["nla_inline", "nla_placebo_inline"])]  # fmt: skip
RUNS = CUTOFF_QA_INLINE_RUNS  # one run per condition, 999 items
GROUPS = {
    "all items": None,
    "stale (Qwen's answer is out of date)": "stable_wrong",
    "not stale (Qwen's answer still true)": "stable_right",
}


def per_item(runs: list[Path], by: str | Callable[[dict], object] = "gate_class") -> dict[str, dict[str, tuple]]:
    """condition -> item -> (the row's `by` field, or by(row); mean accuracy over draws), pooled over the runs."""
    acc: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    cls = {}
    for line in (ln for run in runs for ln in (run / "predictions.jsonl").open()):
        r = json.loads(line)
        acc[r["condition"]][r["item"]].append(bool(r["correct"]))
        cls[r["item"]] = by(r) if callable(by) else r[by]
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


def diff_panel(ax, data: dict, pairs: list = PAIRS, value=None, clusters=None) -> None:
    """A row per pair, grouped by the question it answers: the mean paired difference in per-item accuracy (pp) with its 95%
    bootstrap CI over items, on the items whose `by` is `value` (None: every item)."""
    rows: list = []
    for q, group in pairs:
        rows.append(q)
        for a, b in group:
            keys = [i for i in data[a] if value is None or data[a][i][0] == value]
            ds = [100 * (data[a][i][1] - data[b][i][1]) for i in keys]
            interval = grouped_mean_ci(ds, [clusters[i] for i in keys])[1:] if clusters else ci(ds)
            rows.append((a, b, sum(ds) / len(ds), *interval))
    diff_rows(ax, rows)


def diffs(
    runs: list[Path], out: Path, xlim: tuple[float, float] = (-8, 16), pairs: list = PAIRS, noun: str = "items", data=None, clusters=None
) -> None:
    """graphs.style.diff_rows over every item: filled when the CI excludes 0, labelled by the two conditions' swatches."""
    data = data if data is not None else per_item(runs, by=lambda _: None)  # every item
    n_rows = sum(1 + len(group) for _, group in pairs)
    fig, ax = plt.subplots(figsize=(6.2, 0.35 + 0.34 * n_rows))
    ax.set_xlim(*xlim)
    diff_panel(ax, data, pairs, clusters=clusters)
    ax.set_xlabel(f"difference in % of {noun} right, paired by {noun[:-1]} (95% bootstrap CI; filled = excludes 0)", fontsize=8.5)
    ax.set_title(f"All {len(data['transcript_inline'])} {noun}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(out)


def bar_chart(runs: list[Path], out: Path, by, groups: dict, title: str, ylabel: str = "% of items where Haiku names Qwen's answer",
              bars: list = BARS, panel_w: float = 3.7, data=None, clusters=None, note: str = "") -> None:  # fmt: skip
    """A panel per group of items, a bar per condition: the mean over items of per-item accuracy over draws, 95% bootstrap CIs over
    items. `groups` maps a group's label to a value of the rows' `by` (None: every item)."""
    data = data if data is not None else per_item(runs, by=by)
    xs, ticks = positions(data, bars)
    fig, axes = plt.subplots(1, len(groups), figsize=(panel_w * len(groups), 3.6), sharey=True, layout="constrained")
    top = 0.0
    for ax, (group, value) in zip(axes, groups.items(), strict=True):
        n = 0
        for c, x in xs.items():
            keys = [i for i, (g, _) in data[c].items() if value is None or g == value]
            vals = [data[c][i][1] for i in keys]
            m, lo, hi = grouped_mean_ci(vals, [clusters[i] for i in keys]) if clusters else (sum(vals) / len(vals), *ci(vals))
            ci_bar(ax, x, c, 100 * m, 100 * lo, 100 * hi)
            top, n = max(top, 100 * hi), len(vals)
        ax.set_xticks(ticks, [name for name, cs in bars if any(c in data for c in cs)])
        ax.set_xlim(-0.6, ticks[-1] + 1.0)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{group}, n = {n}", fontsize=9.5)
    axes[0].set_ylim(0, top + 7)  # the axis stops just above the tallest whisker's label
    axes[0].set_ylabel(ylabel)
    fig.suptitle(title, x=0.01, ha="left")
    fig.text(0.995, 0.97, "hatched = placebo", ha="right", fontsize=8, color=MUTED)
    if note:
        fig.supxlabel(note, fontsize=8, color=MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
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
