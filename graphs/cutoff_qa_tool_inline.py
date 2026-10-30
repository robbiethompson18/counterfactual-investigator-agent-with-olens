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

from graphs.cutoff_qa_inline import COLOR, HATCH, INK, LABEL, MUTED, ci, per_item

GROUPS = {"all items": None, "stale\n(answer out of date)": "stable_wrong", "not stale\n(answer still true)": "stable_right"}

SERIES = ["transcript_inline", "olens_inline", "olens_placebo_inline"]
PANELS = {
    "Plain answer: last read 5 tokens before it": Path("data/runs/cfbench/2026-10-29_cutoff-qa-haiku-200"),
    "submit_answer tool call: last read 20 tokens before it": Path("data/runs/cfbench/2026-10-29_cutoff-qa-tool-haiku-200"),
}


def main(out: Path) -> None:
    fig, axes = plt.subplots(1, len(PANELS), figsize=(11, 3.9), sharey=True)
    width, gap = 0.24, 0.015
    for ax, (title, run) in zip(axes, PANELS.items(), strict=True):
        data = per_item([run])
        for i, gate in enumerate(GROUPS.values()):
            for j, s in enumerate(SERIES):
                xs = [a for c, a in data[s].values() if gate is None or c == gate]
                m, (lo, hi) = sum(xs) / len(xs), ci(xs)
                x = i + (j - (len(SERIES) - 1) / 2) * (width + gap)
                ax.bar(x, 100 * m, width, color=COLOR[s], hatch=HATCH.get(s), edgecolor=INK if s in HATCH else None,
                       lw=0, zorder=3)  # fmt: skip
                ax.errorbar(
                    x, 100 * m, yerr=[[100 * (m - lo)], [100 * (hi - m)]], fmt="none", ecolor=MUTED, elinewidth=1, capsize=3, zorder=4
                )
                ax.text(x, 100 * hi + 1.5, f"{100 * m:.0f}", ha="center", va="bottom", fontsize=8, color=INK)
            n = sum(1 for c, _ in data[SERIES[0]].values() if gate is None or c == gate)
            ax.text(i, -0.15, f"n = {n}", ha="center", va="top", fontsize=8, color=MUTED, transform=ax.get_xaxis_transform())
        ax.set_xticks(range(len(GROUPS)))
        ax.set_xticklabels(GROUPS.keys(), fontsize=8.5)
        ax.set_xlim(-0.55, len(GROUPS) - 0.45)
        ax.set_ylim(0, 108)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    axes[0].set_ylabel("% of items where Haiku names Qwen's answer")
    fig.suptitle("Predicting Qwen3.6-27B's answer from the prompt: does moving the answer away from the read hurt the O-lens?",
                 x=0.07, ha="left", fontsize=10.5, color=INK, y=1.07)  # fmt: skip
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR[s], hatch=HATCH.get(s), edgecolor=INK if s in HATCH else None,
                             lw=0) for s in SERIES]  # fmt: skip
    fig.legend(handles, [LABEL[s] for s in SERIES], loc="upper left", frameon=False, fontsize=8.5, ncol=3, bbox_to_anchor=(0.065, 1.02))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("docs/img/cutoff-qa-tool-inline.png"))
    main(p.parse_args().out)
