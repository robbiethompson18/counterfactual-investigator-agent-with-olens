"""Sankey of how a single-turn dataset is filtered into eval cases (the write-up's "Data Filter" section), SuperGPQA first.

    uv run python -m graphs.data_filter_sankey [--out docs/img/data-filter-sankey-supergpqa.png]

Counts are constants from docs/supergpqa-inline.md (Pool and Split): 10,000 non-calculation SuperGPQA questions, 8 Qwen3.6-27B
samples each (thinking off); consistent = the modal bare letter in >= 7 of 8; seed-0 split of the consistent questions into
the 3,000 eval cases and the probe pool. The only filter is consistency: nothing selects on Qwen being right or wrong.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from graphs.style import BG, INK, MUTED

KEPT, DROPPED, WRONG = "#7d7466", "#d9d4c8", "#b5543c"  # neutral flows; WRONG marks the Qwen-wrong slice (not a condition)
# column -> [(node, count)], top to bottom
COLUMNS = [
    [("SuperGPQA questions", 10_000)],
    [("consistent (≥ 7 of 8 samples agree)", 6_328), ("inconsistent, dropped", 3_672)],
    [("eval cases", 3_000), ("probe / validation pool", 3_328)],
    [("Qwen right", 1_878), ("Qwen wrong", 1_122)],
]
FLOWS = [  # (source, target, count, colour)
    ("SuperGPQA questions", "consistent (≥ 7 of 8 samples agree)", 6_328, KEPT),
    ("SuperGPQA questions", "inconsistent, dropped", 3_672, DROPPED),
    ("consistent (≥ 7 of 8 samples agree)", "eval cases", 3_000, KEPT),
    ("consistent (≥ 7 of 8 samples agree)", "probe / validation pool", 3_328, DROPPED),
    ("eval cases", "Qwen right", 1_878, KEPT),
    ("eval cases", "Qwen wrong", 1_122, WRONG),
]
SCALE, GAP, NODE_W = 1 / 10_000, 0.06, 0.018


def layout() -> dict[str, tuple[float, float, float]]:
    """node -> (x, y_top, height); each column's nodes stack down from y = 1 with GAP between them."""
    nodes = {}
    for x, col in enumerate(COLUMNS):
        y = 1.0
        for name, n in col:
            nodes[name] = (float(x), y, n * SCALE)
            y -= n * SCALE + GAP
    return nodes


def band(ax, x0: float, a: tuple[float, float], x1: float, b: tuple[float, float], color: str) -> None:
    """A smooth band from span a = (top, bottom) at x0 to span b at x1 (smoothstep in x)."""
    t = np.linspace(0, 1, 100)
    s = 3 * t**2 - 2 * t**3
    xs = x0 + (x1 - x0) * t
    ax.fill_between(xs, a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, color=color, alpha=0.85, lw=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/img/data-filter-sankey-supergpqa.png")
    args = ap.parse_args()
    nodes = layout()
    out_used = {k: 0.0 for k in nodes}  # how much of each node's height its outgoing flows have used, from the top
    in_used = {k: 0.0 for k in nodes}
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for src, dst, n, color in FLOWS:
        (xs, ys, _), (xd, yd, _), h = nodes[src], nodes[dst], n * SCALE
        a = (ys - out_used[src], ys - out_used[src] - h)
        b = (yd - in_used[dst], yd - in_used[dst] - h)
        out_used[src] += h
        in_used[dst] += h
        band(ax, xs + NODE_W, a, xd, b, color)
    sources = {src for src, *_ in FLOWS}
    counts = {name: n for col in COLUMNS for name, n in col}
    for name, (x, y, h) in nodes.items():
        ax.add_patch(plt.Rectangle((x, y - h), NODE_W, h, color=INK, lw=0))
        right = name not in sources  # a node with no outgoing flow is labelled on its right, over empty space
        muted = "dropped" in name or "pool" in name
        box = None if right or x == 0 else {"facecolor": BG, "edgecolor": "none", "alpha": 0.85, "pad": 2}
        ax.text(x + NODE_W + 0.04 if right else x - 0.04, y - h / 2, f"{name}\n{counts[name]:,}", va="center",
                ha="left" if right else "right", fontsize=8.5, color=MUTED if muted else INK, bbox=box)  # fmt: skip
    ax.set_xlim(-0.75, len(COLUMNS) - 1 + 0.75)
    ax.set_ylim(min(y - h for _, y, h in nodes.values()) - 0.03, 1.03)
    ax.axis("off")
    ax.set_title("SuperGPQA: the only filter is consistency, not Qwen being wrong", color=INK, loc="left")
    fig.patch.set_facecolor(BG)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
