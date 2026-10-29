"""WorkspaceBench families, Qwen3.6-27B: the O-lens vs stock Qwen (asked directly) vs Sonnet as a text-only reader, one
group of three bars per family. Numbers are hand-copied from the "WSB families" section of the sprint doc (2026-10-28);
update the table below when the doc changes. Wrong Bridge has no numbers yet and is commented out.

    uv run python -m graphs.wsb_families [--out docs/img/wsb-families.png]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["sonnet", "olens", "qwen"]
LABEL = {"olens": "Qwen O-lens", "sonnet": "Sonnet, text only", "qwen": "Stock Qwen, asked directly"}
COLOR = {"olens": "#eb6834", "sonnet": "#2a78d6", "qwen": "#3a9a5b"}  # O-lens orange, text-only blue, Qwen outside the four
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True}
)

# family -> (series -> score), caption under the family. A tuple (shown, faint) draws a hollow bar
# behind the solid one for a second pass@ regime.
FAMILIES = {
    "Basic readout\n(personality third)": (
        {"olens": 0.94, "sonnet": 0.54, "qwen": 1.0},
        "lens pass@4 bullets, layer 60\n(0.43 at a random layer)",
    ),
    "Directed modulation\n(obscure)": (
        {"olens": (0.015, 0.337), "sonnet": 0.0, "qwen": 0.80},
        "lens pass@1, hollow = pass@155\nSonnet 0 by filtering\nQwen 0.47 if asked 'generic LM'",
    ),
    "Introspection\n(vector at layer 12)": (
        {"olens": 0.49, "sonnet": 0.0, "qwen": 0.49},
        "pass@1; Sonnet 0 by construction\n(can't read an injected vector)",
    ),
    # "Wrong bridge\n(two-hop trivia)": ({"olens": None, "sonnet": None, "qwen": None}, "numbers pending"),
    "Knowledge cutoff": (
        {"olens": 0.49, "sonnet": 0.0, "qwen": 0.66},
        "lens pass@5 readouts, layer 60\nQwen asked once; Sonnet 0 by filtering",
    ),
}


def main(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    width, gap = 0.26, 0.02
    for i, (family, (scores, caption)) in enumerate(FAMILIES.items()):
        for j, s in enumerate(SERIES):
            x = i + (j - 1) * (width + gap)
            v = scores[s]
            if isinstance(v, tuple):
                v, faint = v
                ax.bar(x, faint, width, color="none", edgecolor=COLOR[s], lw=1.2, zorder=2)
                ax.text(x, faint + 0.015, f"{faint:.2f}", ha="center", va="bottom", fontsize=7, color=MUTED)
            ax.bar(x, v, width, color=COLOR[s], zorder=3)
            ax.text(x, v + 0.015 if v > 0.03 else 0.015, f"{v:.2f}" if v < 1 else "1.0", ha="center", va="bottom", fontsize=8, color=INK)
        ax.text(i, -0.24, caption, ha="center", va="top", fontsize=7, color=MUTED, transform=ax.get_xaxis_transform())
    ax.set_xticks(range(len(FAMILIES)))
    ax.set_xticklabels(FAMILIES.keys(), fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score (fraction correct)")
    ax.grid(axis="x", visible=False)
    ax.set_title("WorkspaceBench families on Qwen3.6-27B: does the O-lens beat just asking?", loc="left", fontsize=11, color=INK, pad=22)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[s]) for s in SERIES]
    ax.legend(handles, [LABEL[s] for s in SERIES], loc="lower left", frameon=False, fontsize=8.5, ncol=3, bbox_to_anchor=(0, 1.0))
    fig.subplots_adjust(bottom=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("docs/img/wsb-families.png"))
    main(p.parse_args().out)
