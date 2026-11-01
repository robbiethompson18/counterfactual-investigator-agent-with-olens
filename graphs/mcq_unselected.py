"""mcq_unselected: Haiku predicting the letter Qwen gives on an unselected MMLU-Pro set, prompt only vs O-lens, J-lens and NLA
with their placebos, beside both standardized probe read positions, for all cases, the cases where
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

from cfbench.catalog import MCQ_UNSELECTED_RUNS
from cfbench.scoring import standard_mcq_comparison
from graphs.cutoff_qa_inline import BARS, bar_chart, diff_panel, per_item
from graphs.style import MUTED

BARS_MCQ = [*BARS, ("Probe\nuser\nend", ["probe_user_end"]), ("Probe\nfinal\ntoken", ["probe_response_start"])]
RUNS = MCQ_UNSELECTED_RUNS
GROUPS = {"all cases": None, "Qwen's letter is the gold": True, "Qwen's letter is wrong": False}
# the difference chart: (question, [(a, b), ...]) groups, each row a − b
PAIRS = [
    ("Lens vs prompt only", [(f"{lens}_inline", "transcript_inline") for lens in ("olens", "jlens", "nla")]),
    ("Does the lens beat its placebo?", [(f"{lens}_inline", f"{lens}_placebo_inline") for lens in ("olens", "jlens", "nla")]),
    ("Does the placebo alone help?", [(f"{lens}_placebo_inline", "transcript_inline") for lens in ("olens", "jlens", "nla")]),
    ("Between lenses", [("jlens_inline", "olens_inline"), ("olens_inline", "nla_inline"), ("jlens_inline", "nla_inline")]),
    (
        "Standardized probe",
        [
            ("probe_response_start", "probe_user_end"),
            ("probe_response_start", "transcript_inline"),
            ("probe_response_start", "jlens_inline"),
        ],
    ),
]


def qwen_right(row: dict) -> bool:
    return row["modal_cf"] == row["gold"]


def diffs(runs: list[Path], out: Path, data=None, clusters=None) -> None:
    """graphs.cutoff_qa_inline's rows, one panel for all cases and one for the Qwen-wrong cases."""
    data = data if data is not None else per_item(runs, by=qwen_right)
    n_rows = sum(1 + len(group) for _, group in PAIRS)
    fig, axes = plt.subplots(1, 2, figsize=(15, 0.4 + 0.34 * n_rows))
    for ax, (title, right) in zip(axes, [("all cases", None), ("Qwen's letter is wrong", False)], strict=True):
        ax.set_xlim(-15, 95)
        diff_panel(ax, data, PAIRS, right, clusters)
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
    data, clusters = standard_mcq_comparison(a.runs, "mmlu")
    title = "MMLU-Pro: historical lens runs and standardized probes on the same questions"
    bar_chart(
        a.runs,
        a.out,
        qwen_right,
        GROUPS,
        title,
        ylabel="% of cases where the letter Qwen gives is named",
        bars=BARS_MCQ,
        panel_w=4.7,
        data=data,
        clusters=clusters,
        note="Probe: nested two-fold, up to 1,000 training questions; 95% question-group CIs. Lens draws: 3; NLA: 1.",
    )
    diffs(a.runs, a.out_diff, data, clusters)
