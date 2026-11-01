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

from graphs.style import BG, INK, MUTED, TOKEN_FONT, color, label, token_text

TOKENS = ["<|im_start|>", "user", "\n", "Which", " of", " the", " jov", "ian", " planets", " have", " rings", "?", "\n", "A", ".",
          " Neptune", "\n", "B", ".", " Uran", "us", "\n", "C", ".", " Saturn", "\n", "D", ".", " all", " of", " the", " above",
          "\n\n", "Answer", " with", " the", " letter", " only", ".", "<|im_end|>", "\n", "<|im_start|>", "assistant", "\n",
          "<think>", "\n\n", "</think>", "\n\n"]  # fmt: skip
PROMPT_READS = {11, 14, 16, 18, 21, 23, 25, 27, 32, 38}  # "?", each option's "." and newline, the final "."
TAIL_READS = set(range(39, 48))
QUESTION, TAIL = "#e3b23c", "#8c2d4a"  # positions, not conditions, so outside the condition palette
LINE_H = 0.078  # axes units per line
READERS = ("olens", "nla", "jlens")  # every lens reads every marked position, a dot each, top down
DOT_W = 0.014


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/img/read-positions-diagram.png")
    args = ap.parse_args()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    to_axes = ax.transAxes.inverted()

    def put(x: float, y: float, text: str, family: str = TOKEN_FONT, fontsize: float = 9.5, **kw) -> float:
        """Draw text at (x, y) and return its real drawn width in axes units."""
        box = ax.text(x, y, text, family=family, fontsize=fontsize, va="center", ha="left", zorder=3, **kw).get_window_extent()
        return to_axes.transform((box.x1, 0))[0] - to_axes.transform((box.x0, 0))[0]

    x, y = 0.02, 0.93
    for i, tok in enumerate(TOKENS):
        group = QUESTION if i in PROMPT_READS else TAIL if i in TAIL_READS else None
        if group:  # the deck's read token: bold in [brackets], a dot per reader in its colour, the read group underlined
            w = put(x, y, f"[{token_text(tok)}]", color=INK, fontweight="bold")
            for k, reader in enumerate(READERS):
                ax.plot(x + w + DOT_W / 2, y + 0.02 * (1 - k), "o", ms=1.9, color=color(reader), zorder=3)
            ax.plot([x + 0.002, x + w - 0.002], [y - 0.03] * 2, color=group, lw=2.2, solid_capstyle="butt", zorder=2)
            w += DOT_W
        else:
            w = put(x, y, token_text(tok), color=MUTED if tok.startswith("<") or tok.strip() == "" else INK)
        if i == len(TOKENS) - 1:
            ax.add_patch(FancyBboxPatch((x - 0.004, y - 0.042), w + 0.008, 0.084, boxstyle="round,pad=0.002,rounding_size=0.01",
                                        facecolor="none", edgecolor=INK, lw=1.2, zorder=4))  # fmt: skip
            ax.annotate("← last token", (x + w + 0.012, y), xytext=(0, 0), textcoords="offset points", ha="left",
                        va="center", fontsize=8, color=INK)  # fmt: skip
        x += w + 0.006 * bool(group)
        if "\n" in tok:  # line breaks where the render has them
            x, y = 0.02, y - LINE_H
    for k, (group, text) in enumerate([(QUESTION, "question reads: punctuation, option newlines, the final '.'"),
                                       (TAIL, "tail reads: the 9 tokens after the question")]):  # fmt: skip
        ax.plot([0.02, 0.045], [0.13 - 0.05 * k] * 2, color=group, lw=2.2, solid_capstyle="butt")
        ax.text(0.055, 0.13 - 0.05 * k, text, va="center", fontsize=8.5, color=INK)
    lx = 0.0325
    for reader in READERS:  # the dots, as in the prompt
        ax.plot(lx, 0.03, "o", ms=4, color=color(reader))
        lx += put(lx + 0.0225, 0.03, f"{label(reader)} reads", family="sans-serif", fontsize=8.5, color=INK) + 0.06
    ax.set_title("Where the metamodels read (MMLU-Pro, as Qwen sees it, thinking off)", loc="left", color=INK)
    fig.patch.set_facecolor(BG)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
