"""cutoff_grader_mentions: how often the cutoff_qa O-lens reads say "grader" or "graded" (\\bgrade[rd]\\b), with Qwen answering in
plain text (A, and the second 1k A2) vs through the submit_answer tool (B). Novel hits only (the word isn't in the prompt within 40
tokens or in the tool's system block, which never says it); scripts/cutoff_grader_mentions.py makes the data. Whiskers are 95%
bootstrap CIs over items, except the items panel (Wilson).

    uv run python -m graphs.cutoff_grader_mentions [--run data/runs/cutoff_grader_mentions/2026-10-30_regex]

Writes docs/img/cutoff-grader-mentions{,-layers,-items}.png.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from graphs.style import INK, MUTED, use

# runs aren't conditions in the palette: two teals for the no-tool runs, magenta for the tool run
RUNS = {"A": ("#147d7d", "no tool"), "A2": ("#7cc4c4", "no tool, second 1k"), "B": ("#b8336a", "submit_answer tool")}
CLASSES = {
    "q_end": "question's ?",
    "instr_punct": 'mid-instruction , and .\n("If you are not sure,\ngive your best guess.")',
    "instr_end": 'final .\n("…just the name.")',
    "im_end": "<|im_end|>",
    "header": "\\n <|im_start|>\nassistant \\n",
    "think": "empty think block",
}
LAYERS = [32, 40, 48, 52, 56, 60]
COL = "nov_graded"
YLAB = '% of O-lens reads saying "grader" / "graded"'


def boot(d: pd.DataFrame, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Rate (ratio of totals) and its 95% CI, resampling items."""
    g = d.groupby("item")[[f"{COL}_sum", f"{COL}_count"]].sum().values
    idx = np.random.default_rng(seed).integers(0, len(g), (n, len(g)))
    s = g[idx].sum(1)
    return (g[:, 0].sum() / g[:, 1].sum(), *np.percentile(s[:, 0] / s[:, 1], [2.5, 97.5]))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def legend(ax, c: pd.DataFrame) -> None:
    n = c.groupby("run").item.nunique()
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col, _ in RUNS.values()]
    ax.legend(handles, [f"{lab} ({n[r]:,} items)" for r, (_, lab) in RUNS.items()], loc="upper right", frameon=False, fontsize=8.5)


def by_position(c: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.4))
    width = 0.26
    for i, cl in enumerate(CLASSES):
        for j, (run, (col, _)) in enumerate(RUNS.items()):
            d = c[(c.run == run) & (c.cls == cl)]
            if d.empty:  # A2 has no think-block reads
                continue
            m, lo, hi = (100 * v for v in boot(d))
            x = i + (j - 1) * width
            ax.bar(x, m, width * 0.95, color=col, zorder=3)
            ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=MUTED, elinewidth=1, capsize=2.5, zorder=4)
            ax.text(x, hi + 0.25, f"{m:.1f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES.values(), fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel(YLAB)
    ax.set_xlabel("read position in the prompt (the same tokens in all three runs)", fontsize=9, color=MUTED)
    fig.suptitle(
        "With the submit_answer tool, the lens says 'graded' ~7x as often, almost all of it mid-instruction\n"
        "(\"…give your best guess. Your answer will be graded on accuracy.\"); the tool's system block never says it",
        x=0.06, ha="left", fontsize=10.5, color=INK,
    )  # fmt: skip
    legend(ax, c)
    fig.tight_layout()
    fig.savefig(out / "cutoff-grader-mentions.png", dpi=160)


def by_layer(c: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for run, (col, lab) in RUNS.items():
        pts = [boot(c[(c.run == run) & (c.cls == "instr_punct") & (c.layer == lay)]) for lay in LAYERS]
        m, lo, hi = (np.array([100 * p[k] for p in pts]) for k in range(3))
        ax.plot(LAYERS, m, "-o", color=col, label=lab, ms=4, zorder=3)
        ax.fill_between(LAYERS, lo, hi, color=col, alpha=0.2, lw=0)
    ax.set_xticks(LAYERS)
    ax.set_xlabel("layer read")
    ax.set_ylabel(YLAB + "\nat the mid-instruction , and .")
    ax.set_title("The tool run says 'graded' more at every layer", loc="left", fontsize=10, color=INK)
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "cutoff-grader-mentions-layers.png", dpi=160)


def items(c: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4))
    hit = c.groupby(["run", "item"])[f"{COL}_sum"].sum().gt(0).groupby("run")
    for j, (run, (col, lab)) in enumerate(RUNS.items()):
        k, n = int(hit.sum()[run]), int(hit.size()[run])
        p, (lo, hi) = 100 * k / n, (100 * v for v in wilson(k, n))
        ax.bar(j, p, 0.7, color=col, zorder=3)
        ax.errorbar(j, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor=MUTED, capsize=3, zorder=4)
        ax.text(j, hi + 1, f"{p:.0f}%", ha="center", fontsize=9, color=INK)
        ax.text(j, -0.13, f"n = {n:,}", ha="center", va="top", fontsize=8, color=MUTED, transform=ax.get_xaxis_transform())
    ax.set_xticks(range(len(RUNS)))
    ax.set_xticklabels([lab.replace(", ", ",\n") for _, lab in RUNS.values()], fontsize=8.5)
    ax.set_ylim(0, 105)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel('% of items with ≥1 read saying "grader" / "graded"')
    ax.set_title("Items where the lens ever says 'graded'", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out / "cutoff-grader-mentions-items.png", dpi=160)


def main(run: Path, out: Path) -> None:
    use()
    c = pd.read_csv(run / "item_counts.csv")
    out.mkdir(parents=True, exist_ok=True)
    by_position(c, out)
    by_layer(c, out)
    items(c, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("data/runs/cutoff_grader_mentions/2026-10-30_regex"))
    ap.add_argument("--out", type=Path, default=Path("docs/img"))
    a = ap.parse_args()
    main(a.run, a.out)
