"""The one chart style (CLAUDE.md); importing this module applies it. A colour per tool, a placebo in its tool's colour with stripes
in the background colour (an open marker where stripes can't show), hatching for placebos and nothing else, and paired-difference
rows labelled by the two conditions' swatches (A − B), so a row reads like the bars it compares (Robbie, 2026-10-30). Conditions
are keyed by canonical names; `canon` maps cfbench/eval condition ids onto them. Ink, grid and background are the Evals page's
(environments/_lib/page.html).
"""

import shutil
import subprocess
import textwrap
from functools import wraps

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, DrawingArea
from matplotlib.patches import Rectangle
from matplotlib.text import Text

INK, MUTED, FAINT, GRID, BG = (
    "#1c1b19",
    "#6f6b62",
    "#a29d92",
    "#e4e0d6",
    "#f6f5f1",
)  # the Evals page's ink, --muted, --faint, grid, background
BLUE, ORANGE, PURPLE, GREEN, GRAY = "#2a78d6", "#eb6834", "#8e44ad", "#2e9e5b", "#898781"
PLACEBO_HATCH = "////"
OUTLINE = "none"  # a bar with no fill, only an ink outline: a reference line's bar, not a tool
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
    "olens_nl_placebo": ("#c9a227", PLACEBO_HATCH, "O-lens (nl) placebo"),
    "probe": ("#11806a", None, "probe"),
    "probe_user_end": ("#78b7aa", None, "probe: user end"),
    "probe_response_start": ("#11806a", None, "probe: final token"),
    "probe_tuned": ("#c2408a", None, "tuned probe"),  # not the page's purple: that is J++ here
    "transcript+probe": ("#78b7aa", None, "probe + transcript"),  # author_blog_drop's nested combiner, the lighter teal
    # chive_inline's reference bars: CHIVE's own Opus 4.8 arms on the same claims (a lighter shade = the same input, another measure)
    "chive_opus_plain": ("#93bbea", None, "CHIVE Opus 4.8, transcript"),
    "chive_opus_cf": ("#a8432f", None, "CHIVE Opus 4.8, resampling"),
    "transcript_told": ("#1f4e79", None, "transcript only, told the model and its cutoff"),  # cutoff_qa_investigators
    "transcript_told_oldprompt": ("#1f4e79", None, "transcript only, told Qwen's cutoff (older prompt)"),  # its canonical set
    "self_answer": (OUTLINE, None, "answers the question itself"),  # cutoff_qa_investigators: the same-cutoff census
    "null": (OUTLINE, None, "edit does nothing"),
    "qwen_logits": (OUTLINE, None, "Qwen's logits (ceiling)"),
}
ALIASES = {"agentic": "transcript", "transcript_agentic": "transcript", "env_diff": "no_transcript", "olens_inline_nl": "olens_nl"}

# matplotlib doesn't search fontconfig's dirs (e.g. a Nix-installed Inter): register fontconfig's Inter files so the charts aren't
# set in the DejaVu fallback
if shutil.which("fc-list"):
    for f in subprocess.run(["fc-list", "Inter", "file"], check=True, capture_output=True, text=True).stdout.split():
        if f.rstrip(":").endswith((".ttf", ".ttc")):
            font_manager.fontManager.addfont(f.rstrip(":"))
plt.rcParams.update(
    {
        "font.family": ["Inter", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "figure.facecolor": BG,
        "figure.titlesize": 11,
        "figure.titleweight": "semibold",
        "axes.facecolor": BG,
        "axes.titlesize": 10.5,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlecolor": INK,
        "axes.labelsize": 10,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": MUTED,
        "ytick.labelcolor": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "hatch.linewidth": 1.8,
        "savefig.dpi": 180,
    }
)


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
    """ax.bar / Rectangle keywords for condition c. The stripes take their colour from edgecolor, not `hatchcolor`: Chrome's PDF
    viewer (PDFium) draws a hatch with edgecolor "none" as a solid bar. lw 0 keeps the outline invisible (2026-10-31)."""
    col, hatch, _ = STYLE[canon(c)]
    if col == OUTLINE:
        return {"facecolor": "none", "edgecolor": INK, "lw": 0.8}
    return {"facecolor": col, "hatch": hatch, "edgecolor": BG if hatch else "none", "lw": 0}


def marker_kw(c: str) -> dict:
    """ax.scatter keywords: filled in the tool's colour, open for a placebo (hatching doesn't show on markers)."""
    col, hatch, _ = STYLE[canon(c)]
    return {"facecolor": "none" if hatch else col, "edgecolor": col}


def positions(bars, groups: list) -> tuple[dict[str, float], list[float]]:
    """x of every bar (a lens and its placebo touch, groups 0.6 apart) and the tick under each group's centre, for `groups` of
    (name, [conditions]); conditions not in `bars` are left out."""
    xs, ticks, x = {}, [], 0.0
    for _, cs in groups:
        cs = [c for c in cs if c in bars]
        if not cs:
            continue
        for i, c in enumerate(cs):
            xs[c] = x + 0.85 * i
        ticks.append(x + 0.85 * (len(cs) - 1) / 2)
        x += 0.85 * (len(cs) - 1) + 1.45
    return xs, ticks


def ci_bar(ax, x: float, c: str, v: float, lo: float, hi: float, width: float = 0.8, fmt: str = "{:.1f}", fontsize: float = 7.5) -> None:
    """Condition c's bar with its gray CI whisker and the value in ink just above it (an offset in points, so it clears the whisker
    whatever the axis's scale). `fontsize`: the value's (a slide figure passes 7.5 * SLIDE_S)."""
    ax.bar(x, v, width, zorder=3, **bar_kw(c))
    ax.errorbar(x, v, [[v - lo], [hi - v]], color=MUTED, lw=0.9, capsize=3, capthick=0.9, zorder=4)
    ax.annotate(fmt.format(v), (x, hi), xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom", fontsize=fontsize, color=INK)


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
        ax.plot(m, y, "o", ms=6, zorder=4, mec=INK, mfc=INK if lo > 0 or hi < 0 else BG)
        ax.text(hi, y, f"   {value_fmt.format(m)}  [{value_fmt.format(lo)}, {value_fmt.format(hi)}]{unit}", va="center", fontsize=8,
                color=INK)  # fmt: skip
        swatch_pair(ax, y, a, b)
        labels.append(text[0] if text else f"{label(a)} − {label(b)}")
    ax.axvline(0, color=MUTED, lw=0.8, zorder=2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(reversed(labels), fontsize=9, color=INK)
    for tick, row in zip(reversed(ax.get_yticklabels()), rows, strict=True):
        if isinstance(row, str):
            tick.set_fontweight("bold")
            tick.set_color(MUTED)
    ax.tick_params(axis="y", length=0, pad=SWATCH_PAD)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="y", visible=False)


def departure(fig, note: str) -> None:
    """The write-up's "Departure from defaults" line, inside the figure rather than under it in the Markdown, so it travels with
    the chart into the deck (Robbie, 2026-10-31): muted, 7pt at write-up size (scaled with font.size on a slide), wrapped to the
    figure's width, as the figure's supxlabel, after any line the chart already has there; the figure grows by the text's height
    so the axes keep their size (margins re-fractioned when there is no layout engine). Constrained layout pins a supxlabel and
    an `outside lower` figure legend to the same edge and reserves the larger of the two, so a chart with the note anchors its
    legend to its bottom axes instead."""
    size = 0.7 * plt.rcParams["font.size"]
    w, h = fig.get_size_inches()
    lines = textwrap.wrap("Departure from defaults: " + note, width=int(w * 72 / (0.5 * size)))
    extra = (1.3 * size * len(lines) + 4) / 72
    if fig.get_layout_engine() is None:
        sp = fig.subplotpars
        frac = extra / (h + extra)
        fig.subplots_adjust(bottom=sp.bottom * (1 - frac) + frac, top=sp.top * (1 - frac) + frac)
    fig.set_size_inches(w, h + extra)
    fig.supxlabel("\n".join(filter(None, [fig.get_supxlabel(), *lines])), x=0.01, ha="left", fontsize=size, color=MUTED, linespacing=1.3)


# Slide figures (slides/talk, made by graphs.talk): the look above at the deck's sizes. The 9pt theme's text is 2/3 of its size
# outside the title slide (Kylie, 2026-10-30), so a figure's 8pt is too. A figure is made at the size it will sit on the slide
# and included unscaled (\slidefig in slides/beamerthemeolens.sty), so its text matches the slide's: vector PDF, transparent
# background. The 16:9 frame's text block is 150 mm wide. SLIDE_RC is applied around one figure by @slide, never process-wide:
# a module that imports a slide figure's statistics must not inherit transparent PNGs and slide-sized text (that is what
# turned the write-up's headline charts white and small on 2026-10-31).
SLIDE_S = 2 / 3
SLIDE_RC: dict = {
    "font.size": 8 * SLIDE_S,
    "axes.titlesize": 8.5 * SLIDE_S,
    "axes.labelsize": 8 * SLIDE_S,
    "xtick.labelsize": 7.5 * SLIDE_S,
    "ytick.labelsize": 7.5 * SLIDE_S,
    "legend.fontsize": 7.5 * SLIDE_S,
    "pdf.fonttype": 42,
    "savefig.transparent": True,
    "savefig.format": "pdf",
}
MM = 1 / 25.4
FULL_W, FULL_H = 150, 58  # a figure that fills the frame under its title
HALF_W = 72  # one column of a two-column frame


def slide(draw):
    """Decorator: run a slide-figure function under SLIDE_RC (text sizes resolve when artists are made, transparency when the
    PDF is saved, so the whole function runs inside it)."""

    @wraps(draw)
    def run(*args, **kwargs):
        with plt.rc_context(SLIDE_RC):
            return draw(*args, **kwargs)

    return run


def slide_figure(w_mm: float = FULL_W, h_mm: float = FULL_H, **kw):
    return plt.subplots(figsize=(w_mm * MM, h_mm * MM), layout="constrained", **kw)


def slide_save(fig, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)  # no bbox_inches="tight": that changes the size the deck relies on
    plt.close(fig)
    print(path)
