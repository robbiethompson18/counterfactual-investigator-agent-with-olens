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

from graphs.cutoff_qa_inline import PAIRS, bar_chart, diffs

RUNS = [Path(f"data/runs/cfbench/2026-10-29_supergpqa-haiku-3k-{s}") for s in ("ts", "olens", "jlens", "nla")]
GROUPS = {"all cases": None, "Qwen wrong (its letter is not the gold one)": "wrong", "Qwen right": "right"}
# cutoff_qa_inline's rows, the last one as NLA − J++
SUPERGPQA_PAIRS = [
    *PAIRS[:-1],
    ("Between lenses", [("olens_inline", "jlens_inline"), ("olens_inline", "nla_inline"), ("nla_inline", "jlens_inline")]),
]


def qwen_right(row: dict) -> str:
    """right | wrong: Qwen's modal letter vs gold."""
    return "right" if row["modal_cf"] == row["gold"] else "wrong"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", default=RUNS)
    p.add_argument("--out", type=Path, default=Path("docs/img/supergpqa-inline.png"))
    p.add_argument("--out-diff", type=Path, default=Path("docs/img/supergpqa-inline-diff.png"))
    a = p.parse_args()
    title = "Predicting Qwen3.6-27B's letter on a SuperGPQA question from the prompt"
    bar_chart(a.runs, a.out, qwen_right, GROUPS, title, ylabel="% of cases where Haiku names Qwen's letter")
    diffs(a.runs, a.out_diff, xlim=(-12, 30), pairs=SUPERGPQA_PAIRS, noun="cases")
