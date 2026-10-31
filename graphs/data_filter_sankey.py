"""Sankeys of how each filtered single-turn dataset becomes eval cases (the write-up's "Data Filter" section). Dark green is
evaluated, Qwen right and wrong alike; light green is usable but held out; gray is unusable. Essay grading has no filter, so no
diagram.

    uv run python -m graphs.data_filter_sankey [--out-dir docs/img]

Counts are constants: docs/supergpqa-inline.md (Pool and Split), docs/mcq-unselected.md (the split, both batches),
docs/envs/cutoff_qa.md and data/banks/cutoff_qa/v3.json (stable = 7 of 8 samples agree; the 1,197 not evaluated include the numeric
items), docs/favorites-qa-inline.md.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from graphs.style import BG, INK, MUTED

EVAL, HELD, UNUSABLE = "#1b5e3b", "#a9d6b8", "#cfcbc0"  # not the NLA's green or the no-tools gray: these aren't conditions
STABLE, UNSTABLE = "consistent (≥ 7 of 8 samples agree)", "inconsistent, dropped"
GAP, NODE_W = 0.06, 0.018
# file slug -> title, columns of (node, count) top to bottom, flows of (source, target, count, colour)
DATASETS = {
    "supergpqa": (
        "SuperGPQA: the only filter is consistency, not Qwen being wrong",
        [
            [("SuperGPQA questions", 10_000)],
            [(STABLE, 6_328), (UNSTABLE, 3_672)],
            [("eval cases", 3_000), ("probe / validation pool", 3_328)],
            [("Qwen right", 1_878), ("Qwen wrong", 1_122)],
        ],
        [
            ("SuperGPQA questions", STABLE, 3_000, EVAL),
            ("SuperGPQA questions", STABLE, 3_328, HELD),
            ("SuperGPQA questions", UNSTABLE, 3_672, UNUSABLE),
            (STABLE, "eval cases", 3_000, EVAL),
            (STABLE, "probe / validation pool", 3_328, HELD),
            ("eval cases", "Qwen right", 1_878, EVAL),
            ("eval cases", "Qwen wrong", 1_122, EVAL),
        ],
    ),
    "mmlu": (
        "MMLU-Pro: two random batches, consistency the only filter",
        [
            [("MMLU-Pro questions", 12_032)],
            [("sampled for eval", 4_000), ("probe half", 2_000), ("not sampled", 6_032)],
            [("eval cases, consistent", 2_523), (UNSTABLE, 1_477)],
            [("Qwen right", 2_065), ("Qwen wrong", 458)],
        ],
        [
            ("MMLU-Pro questions", "sampled for eval", 2_523, EVAL),
            ("MMLU-Pro questions", "sampled for eval", 1_477, UNUSABLE),
            ("MMLU-Pro questions", "probe half", 2_000, HELD),
            ("MMLU-Pro questions", "not sampled", 6_032, HELD),
            ("sampled for eval", "eval cases, consistent", 2_523, EVAL),
            ("sampled for eval", UNSTABLE, 1_477, UNUSABLE),
            ("eval cases, consistent", "Qwen right", 2_065, EVAL),
            ("eval cases, consistent", "Qwen wrong", 458, EVAL),
        ],
    ),
    "cutoff": (
        "Knowledge cutoff: stale and current answers are both kept",
        [
            [("bank v3 questions", 5_467)],
            [(STABLE, 2_196), (UNSTABLE, 3_271)],
            [("eval cases", 999), ("not evaluated", 1_197)],
            [("stale answer", 339), ("current answer", 660)],
        ],
        [
            ("bank v3 questions", STABLE, 999, EVAL),
            ("bank v3 questions", STABLE, 1_197, HELD),
            ("bank v3 questions", UNSTABLE, 3_271, UNUSABLE),
            (STABLE, "eval cases", 999, EVAL),
            (STABLE, "not evaluated", 1_197, HELD),
            ("eval cases", "stale answer", 339, EVAL),
            ("eval cases", "current answer", 660, EVAL),
        ],
    ),
    "favorites": (
        "Idiosyncratic preferences: every consistent question is evaluated",
        [
            [("bank v3 questions", 5_187)],
            [("eval cases, consistent", 1_500), (UNSTABLE, 3_687)],
        ],
        [
            ("bank v3 questions", "eval cases, consistent", 1_500, EVAL),
            ("bank v3 questions", UNSTABLE, 3_687, UNUSABLE),
        ],
    ),
}


def layout(columns: list, scale: float) -> dict[str, tuple[float, float, float]]:
    """node -> (x, y_top, height); each column's nodes stack down from y = 1 with GAP between them."""
    nodes = {}
    for x, col in enumerate(columns):
        y = 1.0
        for name, n in col:
            nodes[name] = (float(x), y, n * scale)
            y -= n * scale + GAP
    return nodes


def band(ax, x0: float, a: tuple[float, float], x1: float, b: tuple[float, float], color: str) -> None:
    """A smooth band from span a = (top, bottom) at x0 to span b at x1 (smoothstep in x)."""
    t = np.linspace(0, 1, 100)
    s = 3 * t**2 - 2 * t**3
    xs = x0 + (x1 - x0) * t
    ax.fill_between(xs, a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, color=color, lw=0)


def sankey(title: str, columns: list, flows: list, out: Path) -> None:
    scale = 1 / columns[0][0][1]
    nodes = layout(columns, scale)
    out_used = {k: 0.0 for k in nodes}  # how much of each node's height its outgoing flows have used, from the top
    in_used = {k: 0.0 for k in nodes}
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for src, dst, n, color in flows:
        (xs, ys, _), (xd, yd, _), h = nodes[src], nodes[dst], n * scale
        a = (ys - out_used[src], ys - out_used[src] - h)
        b = (yd - in_used[dst], yd - in_used[dst] - h)
        out_used[src] += h
        in_used[dst] += h
        band(ax, xs + NODE_W, a, xd, b, color)
    sources = {src for src, *_ in flows}
    evaluated = {dst for _, dst, _, color in flows if color == EVAL}
    counts = {name: n for col in columns for name, n in col}
    for name, (x, y, h) in nodes.items():
        ax.add_patch(plt.Rectangle((x, y - h), NODE_W, h, color=INK, lw=0))
        right = name not in sources  # a node with no outgoing flow is labelled on its right, over empty space
        muted = x > 0 and name not in evaluated
        box = None if right or x == 0 else {"facecolor": BG, "edgecolor": "none", "alpha": 0.85, "pad": 2}
        ax.text(x + NODE_W + 0.04 if right else x - 0.04, y - h / 2, f"{name}\n{counts[name]:,}", va="center",
                ha="left" if right else "right", fontsize=8.5, color=MUTED if muted else INK, bbox=box)  # fmt: skip
    ax.set_xlim(-0.75, len(columns) - 1 + 0.75)
    ax.set_ylim(min(y - h for _, y, h in nodes.values()) - 0.03, 1.03)
    ax.axis("off")
    ax.set_title(title, color=INK, loc="left")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("docs/img"))
    out_dir = ap.parse_args().out_dir
    for slug, (title, columns, flows) in DATASETS.items():
        sankey(title, columns, flows, out_dir / f"data-filter-sankey-{slug}.png")
