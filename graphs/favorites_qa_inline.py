"""favorites_qa_inline: Haiku predicting Qwen's own pick for a free-response preference question, transcript-only vs O-lens,
J++ and the NLA, each with its placebo, one group of seven bars for all items and one per question shape (pick_one, recommend,
choose_for, favorite, name_one; the 7 pilot be_one/least_favorite items count only in "all"). Bars, whiskers, colours and the
paired-difference figure are graphs.cutoff_qa_inline's (docs/favorites-qa-inline.md).

    uv run python -m graphs.favorites_qa_inline [--runs data/runs/cfbench/<run> ...] [--out docs/img/favorites-qa-inline.png]
        [--out-diff docs/img/favorites-qa-inline-diff.png]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from graphs.cutoff_qa_inline import COLOR, HATCH, INK, LABEL, MUTED, SERIES, ci, diffs, per_item

RUNS = [Path(f"data/runs/cfbench/2026-10-29_favorites-haiku-v3-{s}") for s in ("olens", "jlens", "nla")]
GROUPS = {"all items": None, "pick one": "pick_one", "recommend": "recommend", "choose for\na purpose": "choose_for",
          "favorite": "favorite", "name one": "name_one"}  # fmt: skip


def main(runs: list[Path], out: Path) -> None:
    data = per_item(runs, by="shape")
    fig, ax = plt.subplots(figsize=(13, 3.9))
    width, gap = 0.12, 0.008
    for i, (group, shape) in enumerate(GROUPS.items()):
        for j, s in enumerate(SERIES):
            xs = [a for c, a in data[s].values() if shape is None or c == shape]
            m, (lo, hi) = sum(xs) / len(xs), ci(xs)
            x = i + (j - (len(SERIES) - 1) / 2) * (width + gap)
            ax.bar(x, 100 * m, width, color=COLOR[s], hatch=HATCH.get(s), edgecolor=INK if s in HATCH else None,
                   lw=0, zorder=3)  # fmt: skip
            ax.errorbar(x, 100 * m, yerr=[[100 * (m - lo)], [100 * (hi - m)]], fmt="none", ecolor=MUTED, elinewidth=1, capsize=3, zorder=4)
            ax.text(x, 100 * hi + 1.5, f"{100 * m:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK)
        n = sum(1 for c, _ in data[SERIES[0]].values() if shape is None or c == shape)
        ax.text(i, -0.15, f"n = {n}", ha="center", va="top", fontsize=8, color=MUTED, transform=ax.get_xaxis_transform())
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels(GROUPS.keys(), fontsize=9)
    ax.set_xlim(-0.55, len(GROUPS) - 0.45)
    ax.set_ylim(0, 108)
    ax.set_ylabel("% of items where Haiku names Qwen's answer")
    ax.grid(axis="x", visible=False)
    ax.set_title("Predicting Qwen3.6-27B's own pick for a preference question from the prompt", loc="left", fontsize=10.5,
                 color=INK, pad=24)  # fmt: skip
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=COLOR[s], hatch=HATCH.get(s), edgecolor=INK if s in HATCH else None,
                             lw=0) for s in SERIES]  # fmt: skip
    ax.legend(handles, [LABEL[s] for s in SERIES], loc="lower left", frameon=False, fontsize=8.5, ncol=7, bbox_to_anchor=(0, 1.0))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", default=RUNS)
    p.add_argument("--out", type=Path, default=Path("docs/img/favorites-qa-inline.png"))
    p.add_argument("--out-diff", type=Path, default=Path("docs/img/favorites-qa-inline-diff.png"))
    a = p.parse_args()
    main(a.runs, a.out)
    diffs(a.runs, a.out_diff, xlim=(-32, 38))
