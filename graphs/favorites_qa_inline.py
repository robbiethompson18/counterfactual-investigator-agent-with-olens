"""favorites_qa_inline: Haiku predicting Qwen's own pick for a free-response preference question, transcript-only vs O-lens,
J++ and the NLA, each with its placebo, one group of seven bars for all items and one per question shape (pick_one, recommend,
choose_for, favorite, name_one; the 7 pilot be_one/least_favorite items count only in "all"). Bars, whiskers, colours and the
paired-difference figure are graphs.cutoff_qa_inline's (bar_chart, diffs) (docs/favorites-qa-inline.md).

    uv run python -m graphs.favorites_qa_inline [--runs data/runs/cfbench/<run> ...] [--out docs/img/favorites-qa-inline.png]
        [--out-diff docs/img/favorites-qa-inline-diff.png]
"""

import argparse
from pathlib import Path

from graphs.cutoff_qa_inline import bar_chart, diffs

RUNS = [Path(f"data/runs/cfbench/2026-10-29_favorites-haiku-v3-{s}") for s in ("olens", "jlens", "nla")]
GROUPS = {"all items": None, "pick one": "pick_one", "recommend": "recommend", "choose for a purpose": "choose_for",
          "favorite": "favorite", "name one": "name_one"}  # fmt: skip


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", default=RUNS)
    p.add_argument("--out", type=Path, default=Path("docs/img/favorites-qa-inline.png"))
    p.add_argument("--out-diff", type=Path, default=Path("docs/img/favorites-qa-inline-diff.png"))
    a = p.parse_args()
    title = "Predicting Qwen3.6-27B's own pick for a preference question from the prompt"
    bar_chart(a.runs, a.out, "shape", GROUPS, title, panel_w=3.0)
    diffs(a.runs, a.out_diff, xlim=(-32, 38))
