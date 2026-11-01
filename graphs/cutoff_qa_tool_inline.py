"""cutoff_qa_tool_inline: the cutoff_qa_inline chart for Qwen answering in plain text (the last read is 5 tokens before the
answer) next to Qwen answering through a required submit_answer tool call (20 tokens before it), one panel each:
transcript-only vs O-lens vs O-lens placebo for all items, the stale ones and the rest. Bars are the mean over items of the
per-item accuracy over draws (exact match to Qwen's modal answer); whiskers are 95% bootstrap CIs over items. The two panels
are different 200-item sets (117 shared), each labelled by its own gate (docs/cutoff-qa-inline.md).

    uv run python -m graphs.cutoff_qa_tool_inline [--out docs/img/cutoff-qa-tool-inline.png]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from cfbench.catalog import CUTOFF_QA_TOOL_PANELS
from graphs.cutoff_qa_inline import ci, per_item
from graphs.style import MUTED, OLENS, bar_kw, ci_bar, label

GROUPS = {"all items": None, "stale\n(answer out of date)": "stable_wrong", "not stale\n(answer still true)": "stable_right"}

SERIES = ["transcript_inline", "olens_inline", "olens_placebo_inline"]
PANELS = CUTOFF_QA_TOOL_PANELS


def main(out: Path) -> None:
    fig, axes = plt.subplots(1, len(PANELS), figsize=(11, 3.9), sharey=True)
    width, gap, top = 0.24, 0.015, 0.0
    for ax, (title, run) in zip(axes, PANELS.items(), strict=True):
        data = per_item([run])
        for i, gate in enumerate(GROUPS.values()):
            for j, s in enumerate(SERIES):
                xs = [a for c, a in data[s].values() if gate is None or c == gate]
                m, (lo, hi) = sum(xs) / len(xs), ci(xs)
                x = i + (j - (len(SERIES) - 1) / 2) * (width + gap)
                ci_bar(ax, x, s, 100 * m, 100 * lo, 100 * hi, width=width)
                top = max(top, 100 * hi)
            n = sum(1 for c, _ in data[SERIES[0]].values() if gate is None or c == gate)
            ax.text(i, -0.15, f"n = {n}", ha="center", va="top", fontsize=8, color=MUTED, transform=ax.get_xaxis_transform())
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels(GROUPS.keys(), fontsize=8.5)
        ax.set_xlim(-0.55, len(GROUPS) - 0.45)
        ax.grid(axis="x", visible=False)
        ax.set_title(title)
    axes[0].set_ylim(0, top + 7)
    axes[0].set_ylabel("% of items where Haiku names Qwen's answer")
    fig.suptitle(f"Predicting Qwen3.6-27B's answer from the prompt: does moving the answer away from the read hurt the {OLENS}?",
                 x=0.07, ha="left", y=1.07)  # fmt: skip
    handles = [plt.Rectangle((0, 0), 1, 1, **bar_kw(s)) for s in SERIES]
    fig.legend(handles, [label(s) for s in SERIES], loc="upper left", frameon=False, fontsize=8.5, ncol=3, bbox_to_anchor=(0.065, 1.02))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("docs/img/cutoff-qa-tool-inline.png"))
    main(p.parse_args().out)
