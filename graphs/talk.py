"""The sprint talk's figures (slides/talk/talk.tex), PDFs at their slide size in slides/talk/fig/, plus the LaTeX table
snippets the deck \\inputs. Statistics come from cfbench.scoring (the hint evals' bars from the viewer's evals/NAME.json, essay
grading's and the blog deletions' AUROCs from the runs' predictions), the look from graphs.style (@slide: the deck's sizes and a
transparent background around each figure). `make -C slides` runs this first.

    uv run python -m graphs.talk   # -> slides/talk/fig/*.pdf, *.tex
"""

import json
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from matplotlib.colors import to_hex, to_rgb

from cfbench.scoring import (
    AUTHOR_GROUPS,
    AUTHOR_VERSIONS,
    ESSAY_LENSES,
    HF_REPO,
    answer_bars,
    auroc_bars,
    author_bars,
    essay_auroc_bars,
    essay_flips,
)
from graphs.cutoff_grader_mentions import RUNS, at_token
from graphs.olens_mandarin import HAN, bullets_of
from graphs.olens_mandarin import RUN as MANDARIN_RUN
from graphs.style import BG, FAINT, FULL_W, INK, MUTED, SLIDE_S, bar_kw, positions, slide, slide_figure, slide_save

OUT = Path("slides/talk/fig")
# the page's order in groups, each lens with its hatched placebo beside it under one label (Kylie, 2026-10-30); O-lens
# (nl) is left out (Kylie, 2026-10-31).
GROUPS = [("transcript", ["transcript_inline"]), ("O-lens", ["olens_inline", "olens_placebo_inline"]),
          ("probe", ["probe"]), ("tuned\nprobe", ["probe_tuned"])]  # fmt: skip
PANELS = {"after": "hint after the options", "before": "hint first"}


def errors(bars_of):
    """bars_of as error rates (Kylie, 2026-10-30): 1 - % right, lower is better; stars unchanged."""

    def f(name: str) -> tuple[dict, float, int]:
        bars, chance, n = bars_of(name)
        return {c: (1 - v, [1 - hi, 1 - lo], st) for c, (v, (lo, hi), st) in bars.items()}, 1 - chance, n

    return f


def hinted_baseline(bars_of):
    """bars_of with the dotted line at always naming the hinted (correct) letter: right exactly on the knew half, so 50%
    either way (Kylie, 2026-10-30: in place of the 1-in-10 chance line)."""

    def f(name: str) -> tuple[dict, float, int]:
        bars, _, n = bars_of(name)
        return bars, 0.5, n

    return f


def row(axes, evals: dict[str, str], bars_of, chance_label: str, ylabel: str, pct: bool, head: bool, base: float = 0) -> None:
    """One row of the grid, a panel per hint position, in the template's (and the Evals page's answer chart's) style:
    solid bars (placebo hatched, beside its lens), gray 95% CI whisker, the value in ink above it with the stars over
    that, muted group names under the bars, a dotted baseline labelled at the right. `head`: the column titles and case
    counts above the panels. `base`: where the bars and the axis start (Kylie 2026-10-30: 0.5 for AUROC, whose chance
    line is then the axis itself, and 0.4 for % correct)."""
    data = {k: bars_of(name) for k, name in evals.items()}
    for k in evals:
        data[k] = ({c: b for c, b in data[k][0].items() if any(c in cs for _, cs in GROUPS)}, *data[k][1:])
    for ax, (k, (bars, chance, n)) in zip(axes, data.items(), strict=True):
        xs, ticks = positions(bars, GROUPS)
        for c, x in xs.items():
            v, (lo, hi), st = bars[c]
            ax.bar(x, v - base, 0.8, bottom=base, **bar_kw(c))
            ax.errorbar(x, v, [[v - lo], [hi - v]], color=MUTED, lw=0.9, capsize=3, capthick=0.9)
            # offsets in points, so the labels clear the whisker whatever the row's height
            label = f"{v:.1%}" if pct else f"{v:.3f}"
            ax.annotate(
                label, (x, hi), xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom", fontsize=6.5 * SLIDE_S, color=INK
            )
            ax.annotate(
                st, (x, hi), xytext=(0, 6.5), textcoords="offset points", ha="center", va="bottom", fontsize=6.5 * SLIDE_S, color=INK
            )
        ax.axhline(chance, color=FAINT, lw=0.9, ls=(0, (1, 2)), zorder=1)  # off the axis when above it (letter: 90%)
        ax.set_xticks(ticks, [g for g, cs in GROUPS if any(c in bars for c in cs)])
        ax.set_xlim(-0.6, ticks[-1] + 0.6)
        ax.tick_params(axis="x", labelsize=7 * SLIDE_S)
        if head:
            ax.text(0, 1.2, PANELS[k], transform=ax.transAxes, fontsize=8.5 * SLIDE_S, fontweight="semibold", color=INK)
            ax.text(0, 1.08, f"{n:,} cases · legacy selected cohort", transform=ax.transAxes, fontsize=6.5 * SLIDE_S, color=MUTED)
        ax.grid(axis="x", visible=False)
    # the axis stops just above the tallest whisker's labels, not at the next round number (Kylie, 2026-10-30); gridlines
    # every 10% below it
    top = max(hi for d in data.values() for _, (_, hi), _ in d[0].values()) + 0.07
    ticks = np.arange(base, top, 0.1)
    axes[0].set_ylim(base, top)
    axes[0].set_yticks(ticks, [f"{t:.0%}" if pct else f"{t:.1f}" for t in ticks])
    axes[0].set_ylabel(ylabel)
    if base < data[list(evals)[-1]][1] <= top:
        chance = data[list(evals)[-1]][1]
        axes[-1].text(
            1.0, chance, f" {chance_label}", transform=axes[-1].get_yaxis_transform(), fontsize=7 * SLIDE_S, color=MUTED, va="center"
        )


@slide
def grid() -> None:
    """The two questions as rows (did Qwen know it? as AUROC, which letter? as % correct), hint position as columns."""
    knew = {"after": "hint_qwen_dsv4_tail_inline", "before": "hint_qwen_dsv4_before_tail_inline"}
    letter = {"after": "hint_qwen_dsv4_letter_tail_inline", "before": "hint_qwen_dsv4_before_letter_tail_inline"}
    widths = [positions(auroc_bars(n)[0], GROUPS)[1][-1] + 1 for n in knew.values()]
    fig, axes = slide_figure(FULL_W, 60, nrows=2, ncols=2, sharey="row", gridspec_kw={"width_ratios": widths})
    row(axes[0], knew, auroc_bars, "chance", "did Qwen know it?\nAUROC of P(knew)", pct=False, head=True, base=0.5)
    row(axes[1], letter, hinted_baseline(answer_bars), "always the hinted letter", "which letter?\n% correct",
        pct=True, head=False, base=0.4)  # fmt: skip
    axes[0][-1].text(1.0, 1.08, "hatched = placebo", transform=axes[0][-1].transAxes, fontsize=6.5 * SLIDE_S, color=MUTED, ha="right")
    slide_save(fig, OUT / "grid.pdf")


ESSAY_GROUPS = [
    ("transcript", ["transcript_inline"]),
    ("O-lens", ["olens_inline", "olens_placebo_inline"]),
    ("NLA", ["nla_inline", "nla_placebo_inline"]),
    ("J++", ["jlens_inline", "jlens_placebo_inline"]),
    ("transcript\n+ mean", ["transcript_inline+mean"]),
    ("O-lens\n+ mean", ["olens_inline+mean", "olens_placebo_inline+mean"]),
    ("NLA\n+ mean", ["nla_inline+mean", "nla_placebo_inline+mean"]),
    ("J++\n+ mean", ["jlens_inline+mean", "jlens_placebo_inline+mean"]),
    ("probe", ["probe_user_end"]),  # the user-end probe only (Kylie: one probe is easier to explain)
]  # † Kylie: probably cursed (the slide's footnote)


@slide
def essay() -> None:
    """AUROC of P(the score goes up without the opener), in grid()'s style, the dotted line at the regression-to-the-mean null.
    Full slide width: three lenses with placebos, with and without the mean, don't fit Part 1's bar size beside the table."""
    bars, null, n, _ = essay_auroc_bars()
    fig, ax = slide_figure(FULL_W, 24)
    xs, ticks = positions(bars, ESSAY_GROUPS)
    base = 0.5
    for c, x in xs.items():
        v, (lo, hi), st = bars[c]
        ax.bar(x, v - base, 0.8, bottom=base, **bar_kw(c.removesuffix("+mean")))
        ax.errorbar(x, v, [[v - lo], [hi - v]], color=MUTED, lw=0.9, capsize=3, capthick=0.9)
        ax.annotate(
            f"{v:.3f}", (x, hi), xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom", fontsize=6.5 * SLIDE_S, color=INK
        )
        ax.annotate(st, (x, hi), xytext=(0, 6.5), textcoords="offset points", ha="center", va="bottom", fontsize=6.5 * SLIDE_S, color=INK)
    ax.axhline(null, color=FAINT, lw=0.9, ls=(0, (1, 2)), zorder=1)
    ax.text(1.0, null, " bet at\n the mean", transform=ax.get_yaxis_transform(), fontsize=7 * SLIDE_S, color=MUTED, va="center")
    ax.set_xticks(ticks, [g for g, cs in ESSAY_GROUPS if any(c in bars for c in cs)])
    ax.set_xlim(-0.6, ticks[-1] + 0.6)
    ax.tick_params(axis="x", labelsize=7 * SLIDE_S)
    head = f"{n:,} cases · probe: nested 2 × 2 · AUROC over every draw · hatched = placebo"  # one line, clear of the probe's stars
    ax.text(0, 1.08, head, transform=ax.transAxes, fontsize=6.5 * SLIDE_S, color=MUTED)
    ax.grid(axis="x", visible=False)
    top = max(bars[c][1][1] for c in xs) + 0.1
    yt = np.arange(base, top, 0.1)
    ax.set_ylim(base, top)
    ax.set_yticks(yt, [f"{t:.1f}" for t in yt])
    ax.set_ylabel("AUROC of P(score goes up)")
    slide_save(fig, OUT / "essay.pdf")
    essay_tables()


@slide
def essay_diffs() -> None:
    """The paired differences as a forest plot in diff_rows' style (ink CI line and dot, filled when the CI excludes 0, a line
    at 0), compact for the slide's side column: the table above it carries the numbers and swatches. The same essay
    resamples as the table and the stars; the probe is left out so it doesn't set the axis."""
    pairs = essay_auroc_bars()[3]
    short = {"olens_inline": "O-lens", "nla_inline": "NLA", "jlens_inline": "J++"}
    fig, ax = slide_figure(72, 25)
    ticks, labels, y = [], [], 0.0
    for a, b in [(x, other) for x in ESSAY_LENSES for other in (x.replace("_inline", "_placebo_inline"), "transcript_inline")]:
        d, (lo, hi), _ = pairs[a, b]
        ax.plot([lo, hi], [y, y], color=INK, lw=1.2, solid_capstyle="butt", zorder=3)
        ax.plot(d, y, "o", ms=3.5, zorder=4, mec=INK, mfc=INK if lo > 0 or hi < 0 else BG)
        ticks.append(y)
        labels.append(f"{short[a]} − {'transcript' if b == 'transcript_inline' else 'placebo'}")
        y -= 1
    ax.set_yticks(ticks, labels)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=2)
    ax.set_ylim(y + 0.4, 0.6)
    ax.set_xlim(-0.012, 0.035)
    ax.set_xticks([0, 0.01, 0.02, 0.03], ["0", "+0.01", "+0.02", "+0.03"])
    ax.set_xlabel("Δ AUROC, paired 95% CI")
    ax.grid(axis="y", visible=False)
    slide_save(fig, OUT / "essay_diffs.pdf")


def essay_tables() -> None:
    """The essay results slide's tables as LaTeX (fig/essay_pairs.tex) in the theme's macros, and the
    O-lens-vs-placebo flip counts for the footnote (fig/essay_flips.tex)."""
    pairs = essay_auroc_bars()[3]
    name = {"transcript_inline": r"\sw{transcript}\ transcript", "olens_inline": r"\sw{olens}\ O-lens",
            "olens_placebo_inline": r"\swp{olens}\ placebo", "nla_inline": r"\sw{nla}\ NLA", "nla_placebo_inline": r"\swp{nla}\ placebo",
            "jlens_inline": r"\sw{jlens}\ J++", "jlens_placebo_inline": r"\swp{jlens}\ placebo",
            "probe_user_end": r"\sw{probe}\ probe"}  # fmt: skip
    pf = lambda p: "< .001" if p < 1e-3 else f"{p:.3f}".lstrip("0")
    cell = lambda k: f"{pairs[k][0]:+.3f} & {pf(pairs[k][2])}" if k in pairs else "&"
    rows = [f"{name[a]} − {name[b].split(' ', 1)[1]} & {cell((a, b))} & {cell((f'{a}+mean', f'{b}+mean'))}\\\\"
            for a, b in pairs if not a.endswith("+mean") and not a.startswith("probe")]  # fmt: skip
    tab = lambda spec, lines: f"\\begin{{tabular}}{{@{{}}{spec}@{{}}}}\n" + "\n".join(lines) + "\n\\end{tabular}\n"
    head = [r" & \multicolumn{2}{c}{investigator} & \multicolumn{2}{c}{+ bet at mean}\\", r" & $\Delta$ & p & $\Delta$ & p\\"]
    (OUT / "essay_pairs.tex").write_text(tab("lrrrr", [*head, "\\hline", *rows]))
    (OUT / "essay_flips.tex").write_text("{} and {}".format(*essay_flips()))


@slide
def grader() -> None:
    """Two pairs of bars at the "." ending "If you are not sure, give your best guess.", chat template vs the submit_answer tool
    template, on the 863 cutoff_qa items read in both: the share of O-lens reads (six layers) saying "grader"/"graded" (not in the
    prompt), and the share of items with at least one such read, in a lighter shade of the bar's colour (hatching means placebo)."""
    run = Path("runs/cutoff_grader_mentions/2026-10-30_regex")
    if not (Path("data") / run / "novel_hits.jsonl").exists():
        for f in ("item_counts.csv", "novel_hits.jsonl"):
            hf_hub_download(HF_REPO, f"{run}/{f}", repo_type="dataset", local_dir="data")
    groups = [("% of reads", at_token(Path("data") / run, "."), 1.0),
              ("% of items, any read", at_token(Path("data") / run, ".", items=True), 0.5)]  # fmt: skip
    fig, ax = slide_figure(42, 40)
    for g, (_, bars, strength) in enumerate(groups):
        for j, r in enumerate(("A", "B")):
            x, (v, lo, hi) = g * 2.4 + j * 0.85, bars[r]
            ax.bar(x, v, 0.8, color=to_hex(tuple(strength * np.array(to_rgb(RUNS[r][0])) + (1 - strength) * np.array(to_rgb(BG)))))
            ax.errorbar(x, v, [[v - lo], [hi - v]], color=MUTED, lw=0.9, capsize=3, capthick=0.9)
            ax.annotate(
                f"{v:.1%}",
                (x, hi),
                xytext=(0, 1.5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.5 * SLIDE_S,
                color=INK,
            )
    ax.set_xticks([0.425, 2.825], [lab for lab, _, _ in groups])
    ax.set_xlim(-0.6, 3.85)
    ax.tick_params(axis="x", labelsize=7 * SLIDE_S)
    ax.set_ylim(0, 0.85)
    ax.set_yticks(np.arange(0, 0.81, 0.2), [f"{t:.0%}" for t in np.arange(0, 0.81, 0.2)])
    ax.set_ylabel('saying "grader" / "graded"')
    ax.grid(axis="x", visible=False)
    slide_save(fig, OUT / "grader.pdf")


MANDARIN_LENSES = {"old": "Old Qwen O-lens\n(spaces stripped)", "new": "Qwen O-lens\n(fixed)"}


@slide
def mandarin() -> None:
    """Share of Qwen O-lens bullets containing Mandarin, the old lens (bullet's leading space stripped in its RL reward) vs the
    fixed one, on the same greedy reads (graphs.olens_mandarin's bullet panel). Both bars are the O-lens, so both are its orange."""
    path = MANDARIN_RUN / "reads.jsonl"
    if not path.exists():
        hf_hub_download(HF_REPO, str(path.relative_to("data")), repo_type="dataset", local_dir="data")
    reads = [json.loads(line) for line in path.open()]
    fig, ax = slide_figure(42, 40)
    for x, key in enumerate(MANDARIN_LENSES):
        bullets = bullets_of([r[key]["text"] for r in reads])
        hits = sum(bool(HAN.search(b)) for b in bullets)
        share = hits / len(bullets)
        ax.bar(x, share, 0.6, **bar_kw("olens"))
        ax.annotate(f"{share:.1%}\n{hits:,} / {len(bullets):,}", (x, share), xytext=(0, 1.5), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5 * SLIDE_S, color=INK)  # fmt: skip
    ax.set_xticks(range(len(MANDARIN_LENSES)), MANDARIN_LENSES.values())
    ax.set_xlim(-0.6, 1.6)
    ax.tick_params(axis="x", labelsize=7 * SLIDE_S)
    ax.set_ylim(0, 0.65)
    ax.set_yticks(np.arange(0, 0.61, 0.2), [f"{t:.0%}" for t in np.arange(0, 0.61, 0.2)])
    ax.set_ylabel("bullets containing Mandarin")
    ax.grid(axis="x", visible=False)
    slide_save(fig, OUT / "mandarin.pdf")
    (OUT / "mandarin_n.tex").write_text(f"{len(reads):,}")


@slide
def author() -> None:
    """AUROC of P(the deletion is significant) per condition, pooled over both versions; plus the paired differences table
    (fig/author_pairs.tex)."""
    bars, pairs, n = author_bars()
    one = {c: 0 for _, cs in AUTHOR_GROUPS for c in cs}
    width = positions(one, AUTHOR_GROUPS)[1][-1] + 1.2
    fig, ax0 = slide_figure(70, 58)  # the deck's full figure height (graphs.style FULL_H)
    axes = [ax0]
    base = 0.5
    top = max(hi for (_, (_, hi), _) in bars.values()) + 0.06
    for ax, (ver, title) in zip(axes, AUTHOR_VERSIONS.items(), strict=True):
        xs, ticks = positions(one, AUTHOR_GROUPS)
        for c, x in xs.items():
            val, (lo, hi), st = bars[ver, c]
            ax.bar(x, val - base, 0.8, bottom=base, **bar_kw(c))
            ax.errorbar(x, val, [[val - lo], [hi - val]], color=MUTED, lw=0.9, capsize=3, capthick=0.9)
            for text, dy in ((f"{val:.3f}", 1.5), (st, 6.5)):
                ax.annotate(
                    text, (x, hi), xytext=(0, dy), textcoords="offset points", ha="center", va="bottom", fontsize=6.5 * SLIDE_S, color=INK
                )
        ax.set_xticks(ticks, [g for g, _ in AUTHOR_GROUPS])
        ax.set_xlim(-0.6, width - 0.6)
        ax.tick_params(axis="x", labelsize=7 * SLIDE_S)
        ax.set_title(title, fontsize=7 * SLIDE_S, color=INK, loc="left")
        ax.grid(axis="x", visible=False)
    yt = np.arange(base, top, 0.1)
    axes[0].set_ylim(base, top)
    axes[0].set_yticks(yt, [f"{t:.1f}" for t in yt])
    axes[0].set_ylabel("AUROC of P(significant)")
    axes[-1].text(1.0, 1.12, "hatched = placebo", transform=axes[-1].transAxes, fontsize=6.5 * SLIDE_S, color=MUTED, ha="right")
    slide_save(fig, OUT / "author.pdf")
    name = {"transcript_inline": r"\sw{transcript}\ transcript", "olens_inline": r"\sw{olens}\ O-lens",
            "olens_placebo_inline": r"\swp{olens}\ O-lens placebo", "nla_inline": r"\sw{nla}\ NLA",
            "nla_placebo_inline": r"\swp{nla}\ NLA placebo", "transcript+probe": r"\sw{probe}\ probe + transcript"}  # fmt: skip
    pf = lambda p: "< .001" if p < 1e-3 else f"{p:.3f}".lstrip("0")
    rows = [
        f"{name[a]} − {name[c].split(' ', 1)[1]} & {d:+.3f} & [{lo:+.3f}, {hi:+.3f}] & {pf(p)}\\\\"
        for (a, c), (d, (lo, hi), p) in pairs.items()
    ]
    head = [r" & $\Delta$ AUROC & 95\% CI & p\\"]
    (OUT / "author_pairs.tex").write_text("\\begin{tabular}{@{}lrrr@{}}\n" + "\n".join([*head, "\\hline", *rows]) + "\n\\end{tabular}\n")
    (OUT / "author_n.tex").write_text(f"{n:,}")


if __name__ == "__main__":
    grid()
    essay()
    essay_diffs()
    grader()
    mandarin()
    author()
