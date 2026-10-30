"""The example deck's two figures, from docs/supergpqa-inline.md's tables (numbers copied by hand, so this runs without the
data): accuracy per condition for all cases and the Qwen-wrong ones, and the paired differences with their 95% CIs.

    uv run python -m slides.example.figures   # -> slides/example/fig/*.pdf
"""

from pathlib import Path

from graphs.style import INK, MUTED, diff_rows
from slides.figure import BG, HALF_W, bar_kw, figure, save

OUT = Path(__file__).with_name("fig")
CONDS = ["transcript", "olens", "olens_placebo", "jlens", "jlens_placebo", "nla", "nla_placebo"]
# condition -> (all cases, Qwen wrong), % of cases where Haiku names Qwen's letter
ACC = {"transcript": (68.9, 46.2), "olens": (76.9, 57.0), "olens_placebo": (67.6, 46.1), "jlens": (71.1, 49.6),
       "jlens_placebo": (67.5, 44.7), "nla": (71.5, 49.6), "nla_placebo": (68.4, 47.1)}  # fmt: skip
XPOS = [0, 1.55, 2.4, 3.85, 4.7, 6.15, 7.0]  # each lens beside its placebo
FAINT = "#a29d92"  # the Evals page's --faint
DIFFS = [
    "lens − its placebo",
    ("olens", "olens_placebo", 9.3, 7.8, 10.8),
    ("jlens", "jlens_placebo", 3.6, 2.3, 5.0),
    ("nla", "nla_placebo", 3.1, 1.8, 4.3),
    "placebo − transcript only",
    ("olens_placebo", "transcript", -1.4, -2.6, 0.0),
]


def bars() -> None:
    """The Evals page's answer chart (environments/_lib/page.html, Kylie's screenshot 2026-10-30): a lower-case panel title
    with a muted count under it, % gridlines every 20 with the 0% line as the baseline, solid bars, a gray 95% CI whisker,
    the value in ink above it, muted names under the bars, a dotted baseline labelled at the right. Each lens sits beside
    its hatched placebo, one name under the pair. CIs are binomial over cases (the doc gives only the paired ones)."""
    fig, axes = figure(sharey=True, ncols=2)
    for g, (ax, (title, n)) in enumerate(zip(axes, [("all cases", 3000), ("Qwen wrong", 1122)], strict=True)):
        for x, c in zip(XPOS, CONDS, strict=True):
            v = ACC[c][g] / 100
            half = 1.96 * (v * (1 - v) / n) ** 0.5
            ax.bar(x, v, 0.78, **bar_kw(c))
            ax.errorbar(x, v, half, color=MUTED, lw=0.9, capsize=3, capthick=0.9)
            ax.text(x, v + half + 0.02, f"{v:.0%}", ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.axhline(0.626, color=FAINT, lw=0.9, ls=(0, (1, 2)), zorder=1)
        ax.set_xticks([0, 1.975, 4.275, 6.575], ["transcript", "O-lens", "J++", "NLA"])
        ax.tick_params(axis="x", length=0, labelcolor=MUTED)
        ax.text(0, 1.13, title, transform=ax.transAxes, fontsize=8.5, fontweight="semibold", color=INK)
        ax.text(0, 1.04, f"{n:,} cases", transform=ax.transAxes, fontsize=7.5, color=MUTED)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 1)
    axes[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ["0%", "20%", "40%", "60%", "80%", "100%"])
    axes[0].tick_params(axis="y", labelcolor=MUTED)
    axes[1].text(1.0, 0.626, " gold letter", transform=axes[1].get_yaxis_transform(), fontsize=7, color=MUTED, va="center")
    save(fig, OUT / "accuracy.pdf")


def diffs() -> None:
    """The swatches name each row, so the y labels are left empty and the group headers sit inside the plot."""
    fig, ax = figure(HALF_W, 54)
    diff_rows(ax, [r if isinstance(r, str) else (*r, "") for r in DIFFS])
    for t in ax.texts:
        t.set_fontsize(6.5)
    for k, row in enumerate(DIFFS):
        if isinstance(row, str):
            ax.text(
                -3.6,
                len(DIFFS) - 1 - k,
                row,
                fontsize=7,
                fontweight="semibold",
                color=MUTED,
                va="center",
                bbox={"fc": BG, "ec": "none", "pad": 1},
            )
    ax.set_yticklabels([])
    ax.spines.left.set_visible(False)
    ax.set_xlim(-4, 18)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel("difference (pp), 95% CI")
    save(fig, OUT / "diffs.pdf")


if __name__ == "__main__":
    bars()
    diffs()
