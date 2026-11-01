"""Where the metamodels read: one real prompt, every read position coloured by menu (the write-up's "By token position").

    uv run python -m graphs.read_positions_diagram [--out docs/img/read-positions-diagram.png]

The prompt is MMLU-Pro item 10212 as Qwen3.6-27B saw it (thinking-off chat render), tokens and read kinds copied from the
menus_inline inputs (`runs/cfbench_evals/2026-10-30_menus-inline-inputs/cases.jsonl`, `mcq_unselected/nohint~10212`). Menus are
cfbench/evals/menus_inline's: `prompt` = punctuation and option newlines up to the period ending the user turn, `tail` = the nine
tokens after it, `last` = the final one.
"""

import argparse

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from graphs.style import BG, INK, MUTED

TOKENS = ["<|im_start|>", "user", "\n", "Which", " of", " the", " jov", "ian", " planets", " have", " rings", "?", "\n", "A", ".",
          " Neptune", "\n", "B", ".", " Uran", "us", "\n", "C", ".", " Saturn", "\n", "D", ".", " all", " of", " the", " above",
          "\n\n", "Answer", " with", " the", " letter", " only", ".", "<|im_end|>", "\n", "<|im_start|>", "assistant", "\n",
          "<think>", "\n\n", "</think>", "\n\n"]  # fmt: skip
PROMPT_READS = {11, 14, 16, 18, 21, 23, 25, 27, 32, 38}  # "?", each option's "." and newline, the final "."
TAIL_READS = set(range(39, 48))
QUESTION, TAIL = "#e3b23c", "#8c2d4a"  # positions, not conditions, so outside the condition palette
LINE_H = 0.078  # axes units per line


def shown(tok: str) -> str:
    return tok.replace("\n", "↵")


def draw(ax, s: float = 1.0) -> None:
    """The rendered prompt with its read positions on ax (axes coordinates, 0-1); `s` scales the text (the deck passes
    graphs.style.SLIDE_S). The figure must be drawn once already for text extents to be real (ax.figure.canvas.draw())."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    to_axes = ax.transAxes.inverted()
    x, y = 0.02, 0.93
    for i, tok in enumerate(TOKENS):
        color = QUESTION if i in PROMPT_READS else TAIL if i in TAIL_READS else None
        ink = "white" if color == TAIL else MUTED if tok.startswith("<") or tok.strip() == "" else INK
        t = ax.text(x, y, shown(tok), family="monospace", fontsize=9.5 * s, va="center", ha="left", color=ink, zorder=3)
        box = t.get_window_extent()
        w = to_axes.transform((box.x1, 0))[0] - to_axes.transform((box.x0, 0))[0]  # the token's real drawn width
        if color:
            ax.add_patch(FancyBboxPatch((x + 0.001, y - 0.028), w - 0.002, 0.056, boxstyle="round,pad=0.002,rounding_size=0.008",
                                        facecolor=color, edgecolor="none", alpha=0.9, zorder=2))  # fmt: skip
        if i == len(TOKENS) - 1:
            ax.add_patch(FancyBboxPatch((x - 0.004, y - 0.038), w + 0.008, 0.076, boxstyle="round,pad=0.002,rounding_size=0.01",
                                        facecolor="none", edgecolor=INK, lw=1.4, zorder=4))  # fmt: skip
            ax.annotate("← last token", (x + w + 0.01, y), xytext=(0, 0), textcoords="offset points", ha="left",
                        va="center", fontsize=8 * s, color=INK)  # fmt: skip
        x += w
        if "\n" in tok:  # line breaks where the render has them
            x, y = 0.02, y - LINE_H
    for k, (color, label) in enumerate([(QUESTION, "question reads: punctuation, option newlines, the final '.'"),
                                        (TAIL, "tail reads: the 9 tokens after the question")]):  # fmt: skip
        ax.add_patch(FancyBboxPatch((0.02, 0.07 - 0.055 * k - 0.016), 0.025, 0.032, boxstyle="round,pad=0.003", facecolor=color,
                                    edgecolor="none"))  # fmt: skip
        ax.text(0.055, 0.07 - 0.055 * k, label, va="center", fontsize=8.5 * s, color=INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/img/read-positions-diagram.png")
    args = ap.parse_args()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    draw(ax)
    ax.set_title("Where the metamodels read (MMLU-Pro, as Qwen sees it, thinking off)", loc="left", color=INK)
    fig.patch.set_facecolor(BG)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
