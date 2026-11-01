"""Essay grading's charts for the final write-up ("Essay Grading" in docs/final-writeup.md), the same chart as the deck's essay
slide: AUROC of P(the score goes up with the base opener), transcript only, the three lenses each beside its hatched placebo, the
same seven with Qwen's observed mean grade added ("+ mean", an out-of-fold logistic regression) and the user-end probe, the dotted
line at betting on regression to the mean. `essay_bars(ax, s)` draws it on any axes: here at write-up size as a PNG, in
graphs.talk under @slide as the deck's PDF, so the two never drift. Statistics: cfbench.scoring.essay_auroc_bars (AUROC over every
draw, 95% CI resampling essays, stars: one-sided paired bootstrap vs transcript only). AUROC, not % right: the investigator gives a
probability on a 54% base rate, and % right at 0.5 rewards a shift in its bias (slides/CLAUDE.md's rule).

    uv run python -m graphs.essay_grading [--out-dir docs/img]

Writes headline-essay.png (transcript only, the lenses and the probe), headline-essay-mean.png (the "+ mean" bars and the probe;
the write-up shows the two apart, the deck draws every group on one axes) and headline-essay-diff.png (paired differences: each
lens minus transcript only and minus its placebo, both probes minus transcript only), and prints the differences the write-up
quotes.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cfbench.scoring import ESSAY_LENSES, essay_auroc_bars
from graphs.style import FAINT, INK, MUTED, OLENS, bar_kw, departure, diff_rows, positions

GROUPS = [
    ("transcript", ["transcript_inline"]),
    (OLENS, ["olens_inline", "olens_placebo_inline"]),
    ("NLA", ["nla_inline", "nla_placebo_inline"]),
    ("J++", ["jlens_inline", "jlens_placebo_inline"]),
    ("transcript\n+ mean", ["transcript_inline+mean"]),
    (f"{OLENS}\n+ mean", ["olens_inline+mean", "olens_placebo_inline+mean"]),
    ("NLA\n+ mean", ["nla_inline+mean", "nla_placebo_inline+mean"]),
    ("J++\n+ mean", ["jlens_inline+mean", "jlens_placebo_inline+mean"]),
    ("probe", ["probe_user_end"]),  # the user-end probe only (Kylie: one probe is easier to explain)
]  # † Kylie: probably cursed (the slide's footnote)
PLAIN_GROUPS = [g for g in GROUPS if "+ mean" not in g[0]]  # the write-up's first chart
MEAN_GROUPS = [g for g in GROUPS if "+ mean" in g[0] or g[0] == "probe"]  # its second, under the calibration paragraph
TS = "transcript_inline"
DEPARTURE = "placebo is another opener's reads on a different essay of the same topic; P(up) scored as AUROC."
WRITEUP_S = 1.15  # text scale at write-up size: values at 7.5pt as graphs.style.ci_bar; the deck passes SLIDE_S
BASE = 0.5  # the axis starts at chance


def essay_bars(
    ax,
    s: float,
    headroom: float = 0.1,
    ylabel: str = "AUROC of P(score goes up)",
    mean: bool = True,
    null_line: bool = True,
    groups: list | None = None,
) -> None:
    """The bars on ax, text sizes scaled by s. Bars from chance, the gray 95% CI whisker, the value in ink above it and the stars
    over that (offsets in points, so they clear the whisker at any height), the null dotted and labelled at the right. The axis
    stops `headroom` above the tallest whisker: room for its two labels, so 0.1 on the deck's 24 mm figure, less when taller.
    `ylabel`: the deck's 24 mm figure fits only a short one (the full one clipped to "AUROC of P(score goes u").
    `mean=False` leaves out the "+ mean" groups and `null_line=False` the dotted null (the deck drops both, Kylie 2026-11-01);
    `groups` names which of GROUPS to draw instead (the write-up's two charts)."""
    bars, null, n, _ = essay_auroc_bars()
    if groups is None:
        groups = [(g, cs) for g, cs in GROUPS if mean or not g.endswith("+ mean")]
    xs, ticks = positions(bars, groups)
    for c, x in xs.items():
        v, (lo, hi), st = bars[c]
        ax.bar(x, v - BASE, 0.8, bottom=BASE, **bar_kw(c.removesuffix("+mean")))
        ax.errorbar(x, v, [[v - lo], [hi - v]], color=MUTED, lw=0.9, capsize=3, capthick=0.9)
        ax.annotate(f"{v:.3f}", (x, hi), xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom", fontsize=6.5 * s, color=INK)
        ax.annotate(st, (x, hi), xytext=(0, 6.5), textcoords="offset points", ha="center", va="bottom", fontsize=6.5 * s, color=INK)
    if null_line:
        ax.axhline(null, color=FAINT, lw=0.9, ls=(0, (1, 2)), zorder=1)
        ax.text(1.0, null, " bet at\n the mean", transform=ax.get_yaxis_transform(), fontsize=7 * s, color=MUTED, va="center")
    ax.set_xticks(ticks, [g for g, cs in groups if any(c in bars for c in cs)])
    ax.set_xlim(-0.6, ticks[-1] + 0.6)
    ax.tick_params(axis="x", labelsize=7 * s)
    head = f"{n:,} cases · probe: nested 2 × 2 · AUROC over every draw · hatched = placebo"  # one line, clear of the probe's stars
    ax.text(0, 1.08, head, transform=ax.transAxes, fontsize=6.5 * s, color=MUTED)
    ax.grid(axis="x", visible=False)
    top = max(bars[c][1][1] for c in xs) + headroom
    yt = np.arange(BASE, top, 0.1)
    ax.set_ylim(BASE, top)
    ax.set_yticks(yt, [f"{t:.1f}" for t in yt])
    ax.set_ylabel(ylabel)


def essay_diffs(ax) -> None:
    """The paired differences as diff_rows: each lens minus transcript only, each lens minus its placebo, both probes minus
    transcript only, in AUROC."""
    pairs = essay_auroc_bars()[3]
    rows = []
    for a, b in [*((x, TS) for x in ESSAY_LENSES), *((x, x.replace("_inline", "_placebo_inline")) for x in ESSAY_LENSES),
                 ("probe_user_end", TS), ("probe_response_start", TS)]:  # fmt: skip
        d, (lo, hi), _ = pairs[a, b]
        rows.append((a, b, d, lo, hi))
    print("essay AUROC:", {f"{a[:5]}-{b[:5]}": f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]" for a, b, d, lo, hi in rows})
    diff_rows(ax, rows, value_fmt="{:+.3f}")
    ax.set_xlim(-0.03, 0.2)
    ax.set_xlabel("difference in AUROC, paired by essay (95% bootstrap CI; filled = excludes 0)", fontsize=8.5)


def main(out: Path) -> None:
    charts = [("headline-essay.png", PLAIN_GROUPS, "Essay grading"),
              ("headline-essay-mean.png", MEAN_GROUPS, "Essay grading, with Qwen's observed mean grade added")]  # fmt: skip
    for name, groups, title in charts:
        fig, ax = plt.subplots(figsize=(6.4, 3.2), layout="constrained")
        essay_bars(ax, WRITEUP_S, headroom=0.04, groups=groups)
        fig.suptitle(title, x=0.01, ha="left")
        departure(fig, DEPARTURE)
        fig.savefig(out / name, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out / name)
    fig, ax = plt.subplots(figsize=(7.4, 3.4), layout="constrained")
    essay_diffs(ax)
    fig.savefig(out / "headline-essay-diff.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / "headline-essay-diff.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("docs/img"))
    main(ap.parse_args().out_dir)
