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
from pathlib import Path

import matplotlib.pyplot as plt

from graphs.cutoff_qa_inline import BARS, bar_chart, diff_panel, per_item
from graphs.style import MUTED

BARS_MCQ = [*BARS[:3], ("NLA\n(1 draw)", BARS[3][1]), ("probe", ["probe"]), ("Qwen's\nlogits", ["qwen_logits"])]
RUNS = [Path(f"data/runs/cfbench/2026-10-29_mcq-unselected-haiku-{s}") for s in ("b1r", "b2", "jlens-b1", "jlens-b2", "nla-b1", "nla-b2")]
RUNS.append(Path("data/runs/mcq_answer_probe/2026-10-29_mcq-unselected-probe"))
GROUPS = {"all cases": None, "Qwen's letter is the gold": True, "Qwen's letter is wrong": False}
# the difference chart: (question, [(a, b), ...]) groups, each row a − b
PAIRS = [
    ("Lens vs prompt only", [(f"{lens}_inline", "transcript_inline") for lens in ("olens", "jlens", "nla")]),
    ("Does the lens beat its placebo?", [(f"{lens}_inline", f"{lens}_placebo_inline") for lens in ("olens", "jlens", "nla")]),
    ("Does the placebo alone help?", [(f"{lens}_placebo_inline", "transcript_inline") for lens in ("olens", "jlens", "nla")]),
    ("Between lenses", [("jlens_inline", "olens_inline"), ("olens_inline", "nla_inline"), ("jlens_inline", "nla_inline")]),
    ("What a probe reads from the same activations", [("probe", "transcript_inline"), ("probe", "jlens_inline")]),
]


def qwen_right(row: dict) -> bool:
    return row["modal_cf"] == row["gold"]


def diffs(runs: list[Path], out: Path) -> None:
    """graphs.cutoff_qa_inline's rows, one panel for all cases and one for the Qwen-wrong cases."""
    data = per_item(runs, by=qwen_right)
    n_rows = sum(1 + len(group) for _, group in PAIRS)
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.4 + 0.34 * n_rows))
    for ax, (title, right) in zip(axes, [("all cases", None), ("Qwen's letter is wrong", False)], strict=True):
        ax.set_xlim(-15, 95)
        diff_panel(ax, data, PAIRS, right)
        n = sum(1 for r, _ in data["transcript_inline"].values() if right is None or r == right)
        ax.set_title(f"{title} (n = {n})")
    fig.supxlabel("difference in % of cases right, paired by case (95% bootstrap CI; filled = excludes 0)", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", default=RUNS)
    p.add_argument("--out", type=Path, default=Path("docs/img/mcq-unselected.png"))
    p.add_argument("--out-diff", type=Path, default=Path("docs/img/mcq-unselected-diff.png"))
    a = p.parse_args()
    title = "Predicting Qwen3.6-27B's MMLU-Pro letter from the bare prompt (Haiku 5.5, 3 draws; unselected set, both batches)"
    bar_chart(
        a.runs, a.out, qwen_right, GROUPS, title, ylabel="% of cases where the letter Qwen gives is named", bars=BARS_MCQ, panel_w=4.3
    )
    diffs(a.runs, a.out_diff)
