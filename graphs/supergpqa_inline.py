"""supergpqa_inline: Haiku predicting Qwen's letter on a SuperGPQA question from the prompt, transcript-only vs O-lens, J++ and
the NLA, each with its placebo: one group of seven bars for all cases, for the cases Qwen gets wrong (its modal letter is
not the gold one) and for the ones it gets right. graphs/cutoff_qa_inline.py's chart with the groups changed. Bars are the
mean over cases of the per-case accuracy over draws (the letter matches Qwen's modal letter); whiskers are 95% bootstrap CIs
over cases. The second figure shows the paired differences between conditions with their 95% bootstrap CIs, all cases.
Reads the runs' predictions.jsonl, pooled (docs/supergpqa-inline.md).

    uv run python -m graphs.supergpqa_inline [--runs data/runs/cfbench/<run> ...] [--out docs/img/supergpqa-inline.png]
        [--out-diff docs/img/supergpqa-inline-diff.png]
"""

import argparse
from pathlib import Path

from cfbench.catalog import SUPERGPQA_INLINE_RUNS
from cfbench.scoring import standard_mcq_comparison
from graphs.cutoff_qa_inline import PAIRS, bar_chart, diffs
from graphs.mcq_unselected import BARS_MCQ

RUNS = SUPERGPQA_INLINE_RUNS
GROUPS = {"all cases": None, "Qwen wrong (its letter is not the gold one)": False, "Qwen right": True}
# cutoff_qa_inline's rows, the last one as NLA − J++
SUPERGPQA_PAIRS = [
    *PAIRS[:-1],
    ("Between lenses", [("olens_inline", "jlens_inline"), ("olens_inline", "nla_inline"), ("nla_inline", "jlens_inline")]),
]


def qwen_right(row: dict) -> bool:
    """Qwen's modal letter vs gold."""
    return row["modal_cf"] == row["gold"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", default=RUNS)
    p.add_argument("--out", type=Path, default=Path("docs/img/supergpqa-inline.png"))
    p.add_argument("--out-diff", type=Path, default=Path("docs/img/supergpqa-inline-diff.png"))
    a = p.parse_args()
    data, clusters = standard_mcq_comparison(a.runs, "supergpqa")
    title = "SuperGPQA: historical lens runs and standardized probes on the same questions"
    bar_chart(
        a.runs,
        a.out,
        qwen_right,
        GROUPS,
        title,
        ylabel="% naming Qwen's letter",
        bars=BARS_MCQ,
        panel_w=4.7,
        data=data,
        clusters=clusters,
        note="Probe: nested two-fold, up to 1,000 training questions; 95% question-group CIs. Historical lens draws: 3.",
    )
    diffs(
        a.runs,
        a.out_diff,
        xlim=(-12, 55),
        pairs=[
            *SUPERGPQA_PAIRS,
            ("Standardized probe", [("probe_response_start", "probe_user_end"), ("probe_response_start", "transcript_inline")]),
        ],
        noun="cases",
        data=data,
        clusters=clusters,
    )
