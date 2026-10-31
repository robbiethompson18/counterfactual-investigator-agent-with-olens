"""Blog sentence deletion: text-only Haiku, Haiku with O-lens, and the standalone L2 probe.

    uv run python -m graphs.author_blog_drop

Reported AUROCs and blogger-bootstrap 95% CIs from runs/author_mcq_blog/2026-10-31_period-probe/results.md
in the HF dataset Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens,
revision 0c62b2425c18cfd1f641b25f20b4494bc20cd2b8. Haiku rows: the first table; probe: standard L2, not PCA or ensemble.
Constants retain the report's three-decimal precision. Renders published summaries without model calls.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from graphs.style import MUTED, ci_bar

ROWS = [
    ("Haiku\ntext only", "transcript", 0.797, 0.773, 0.819),
    ("Haiku\nwith O-lens", "olens", 0.803, 0.779, 0.824),
    ("Probe\nalone", "probe", 0.784, 0.760, 0.808),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.1))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.80, bottom=0.23)
    for i, (_, condition, value, lo, hi) in enumerate(ROWS):
        ci_bar(ax, i, condition, value, lo, hi, width=0.64, fmt="{:.3f}")
    ax.set_xticks(range(len(ROWS)), [row[0] for row in ROWS], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("AUROC")
    ax.grid(axis="x", visible=False)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls="--", zorder=4)
    ax.text(2.55, 0.515, "chance", fontsize=7, color=MUTED, ha="right")
    fig.text(0.14, 0.94, "Blog sentence deletion", fontsize=13, weight="bold")
    fig.text(0.14, 0.875, "Predicting a substantial drop in Qwen’s confidence", fontsize=9, color=MUTED)
    fig.text(0.14, 0.09, "1,316 cases · 520 bloggers · 95% bootstrap CIs", fontsize=8, color=MUTED)
    fig.text(0.14, 0.045, "Standalone probe; held-out bloggers. Exploratory results.", fontsize=8, color=MUTED)
    out = Path("docs/img/author-blog-drop.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
