"""The one chart style (CLAUDE.md): a colour per tool, a placebo in its tool's colour with black hatching (an open marker where
hatching can't show), and paired-difference rows labelled by the two conditions' swatches (A − B), so a row reads like the bars
it compares (Robbie, 2026-10-30). Conditions are keyed by canonical names; `canon` maps cfbench/eval condition ids onto them.
"""

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Rectangle
from matplotlib.text import Text

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
BLUE, ORANGE, PURPLE, GREEN, GRAY = "#2a78d6", "#eb6834", "#8e44ad", "#2e9e5b", "#898781"
PLACEBO_HATCH = "////"
# canonical condition -> (colour, hatch, label)
STYLE = {
    "transcript": (BLUE, None, "transcript only"),
    "olens": (ORANGE, None, "O-lens"),
    "olens_placebo": (ORANGE, PLACEBO_HATCH, "O-lens placebo"),
    "jlens": (PURPLE, None, "J++"),
    "jlens_placebo": (PURPLE, PLACEBO_HATCH, "J++ placebo"),
    "nla": (GREEN, None, "NLA"),
    "nla_placebo": (GREEN, PLACEBO_HATCH, "NLA placebo"),
    "no_transcript": (GRAY, None, "no transcript"),
    # the hint evals' extra bars, in the Evals page's colours (page.html --c-olens-nl, --c-probe, --c-probe-tuned)
    "olens_nl": ("#c9a227", None, "O-lens (nl)"),
    "probe": ("#11806a", None, "probe"),
    "probe_tuned": ("#c2408a", None, "tuned probe"),  # not the page's purple: that is J++ here
    "null": ("#ffffff", PLACEBO_HATCH, "edit does nothing"),
}
ALIASES = {"agentic": "transcript", "transcript_agentic": "transcript", "env_diff": "no_transcript", "olens_inline_nl": "olens_nl"}


def canon(c: str) -> str:
    """'olens_placebo_inline' / 'olens_agentic' / 'olens_score' -> 'olens_placebo' / 'olens' / 'olens'."""
    for suffix in ("_inline", "_agentic", "_score"):
        c = c.removesuffix(suffix)
    return ALIASES.get(c, c)


def color(c: str) -> str:
    return STYLE[canon(c)][0]


def label(c: str) -> str:
    return STYLE[canon(c)][2]


def bar_kw(c: str) -> dict:
    """ax.bar / Rectangle keywords for condition c."""
    col, hatch, _ = STYLE[canon(c)]
    return {"facecolor": col, "hatch": hatch, "edgecolor": INK if hatch else "none", "lw": 0.8 if col == "#ffffff" else 0}


def marker_kw(c: str) -> dict:
    """ax.scatter keywords: filled in the tool's colour, open for a placebo (hatching doesn't show on markers)."""
    col, hatch, _ = STYLE[canon(c)]
    return {"facecolor": "none" if hatch else col, "edgecolor": col}


def use() -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "axes.axisbelow": True})  # fmt: skip


SW_W, SW_H, SW_MINUS, SW_GAP = 14, 10, 12, 8  # swatch width, height, the gap holding the minus sign, gap to the axis (points)


def swatch_pair(ax, y: float, a: str, b: str) -> None:
    """A's and B's bar swatches with a minus between them, just left of the y axis at data y (sized in points, so they survive
    tight_layout); set the y tick label pad to SWATCH_PAD to make room."""
    da = DrawingArea(2 * SW_W + SW_MINUS, SW_H)
    da.add_artist(Rectangle((0, 0), SW_W, SW_H, **bar_kw(a)))
    da.add_artist(Rectangle((SW_W + SW_MINUS, 0), SW_W, SW_H, **bar_kw(b)))
    da.add_artist(Text(SW_W + SW_MINUS / 2, SW_H / 2, "−", ha="center", va="center", fontsize=9, color=INK))
    ax.add_artist(AnnotationBbox(da, (0, y), xycoords=ax.get_yaxis_transform(), box_alignment=(1, 0.5), xybox=(-SW_GAP, 0),
                                 boxcoords="offset points", frameon=False, pad=0, annotation_clip=False))  # fmt: skip


SWATCH_PAD = 2 * SW_GAP + 2 * SW_W + SW_MINUS


def diff_rows(ax, rows: list, unit: str = "", value_fmt: str = "{:+.1f}") -> None:
    """Paired-difference rows on ax, top to bottom. A row is a header string, or (a, b, mean, lo, hi[, text]) for A − B with its
    95% CI: an ink CI line and dot, filled when the CI excludes 0, the numbers to the right, and the label (default
    "A − B") followed by the two swatches. One ink for every mark: the swatches say which conditions."""
    labels = []
    for k, row in enumerate(rows):
        y = len(rows) - 1 - k
        if isinstance(row, str):  # a group header
            labels.append(row)
            continue
        a, b, m, lo, hi, *text = row
        ax.plot([lo, hi], [y, y], color=INK, lw=1.6, solid_capstyle="butt", zorder=3)
        ax.plot(m, y, "o", ms=6, zorder=4, mec=INK, mfc=INK if lo > 0 or hi < 0 else "white")
        ax.text(hi, y, f"   {value_fmt.format(m)}  [{value_fmt.format(lo)}, {value_fmt.format(hi)}]{unit}", va="center", fontsize=8,
                color=INK)  # fmt: skip
        swatch_pair(ax, y, a, b)
        labels.append(text[0] if text else f"{label(a)} − {label(b)}")
    ax.axvline(0, color=MUTED, lw=0.8, zorder=2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(reversed(labels), fontsize=9)
    for tick, row in zip(reversed(ax.get_yticklabels()), rows, strict=True):
        if isinstance(row, str):
            tick.set_fontweight("bold")
            tick.set_color(MUTED)
    ax.tick_params(axis="y", length=0, pad=SWATCH_PAD)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="y", visible=False)
