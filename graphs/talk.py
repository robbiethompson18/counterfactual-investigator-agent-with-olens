"""The sprint talk's figures (slides/talk/talk.tex), PDFs at their slide size in slides/talk/fig/, plus the LaTeX table
snippets the deck \\inputs. Statistics come from cfbench.scoring (the hint evals' bars from the viewer's evals/NAME.json, essay
grading's and the blog deletions' AUROCs from the runs' predictions; the essay bars themselves are graphs.essay_grading's, drawn
here at slide size), the look from graphs.style (@slide: the deck's sizes and a
transparent background around each figure). The write-up's dataset parts (docs/final-writeup.md Part 1) are drawn here at slide
size from their write-up modules: the Sankeys (graphs.data_filter_sankey), the headline and per-task bars (graphs.headline_inline)
and the knowledge-cutoff investigators (graphs.cutoff_qa_investigators). `make -C slides` runs this first.

    uv run python -m graphs.talk   # -> slides/talk/fig/*.pdf, *.tex
"""

import json
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import Rectangle

from cfbench.catalog import CF, CUTOFF_GRADER_MENTIONS_RUN, DRINK_RUN, MANDARIN_RUN, MENUS, SNEEZE_RUN
from cfbench.scoring import (
    AUTHOR_GROUPS,
    AUTHOR_VERSIONS,
    ESSAY_LENSES,
    HF_REPO,
    acting_by_factor,
    answer_bars,
    auroc_bars,
    author_bars,
    essay_auroc_bars,
    essay_flips,
)
from graphs import author_blog_drop, cutoff_qa_investigators, data_filter_sankey, headline_inline, hint_unselected
from graphs.cutoff_grader_mentions import RUNS, at_token, wilson
from graphs.essay_grading import DEPARTURE as ESSAY_DEPARTURE
from graphs.essay_grading import essay_bars
from graphs.headline_inline import LENSES, PLACEBOS, POOLED, PROBE, TS, Boot
from graphs.olens_mandarin import HAN, bullets_of
from graphs.style import (
    BG,
    FAINT,
    FULL_W,
    INK,
    MUTED,
    OLENS,
    SLIDE_S,
    bar_kw,
    departure,
    diff_rows,
    label,
    positions,
    slide,
    slide_figure,
    slide_save,
)

GRID_DEPARTURE = (f"left column: investigator Gemini 3.8 Flash at effort low, one draw; both legacy columns: {OLENS} at layers "
                  "32/40/48/52, positions the option newlines and the end of the prompt only; middle: the same prompts and reads, "
                  "Haiku 5.5 at medium, 3 draws pooled; the unselected cohort is on the defaults.")  # fmt: skip

OUT = Path("slides/talk/fig")
# the page's order in groups, each lens with its hatched placebo beside it under one label (Kylie, 2026-10-30); O-lens
# (nl) is left out (Kylie, 2026-10-31).
GROUPS = [("transcript", ["transcript_inline"]), (OLENS, ["olens_inline", "olens_placebo_inline"]),
          ("probe", ["probe"]), ("tuned\nprobe", ["probe_tuned"])]  # fmt: skip
# the hinted-MCQ grid's columns (Robbie, 2026-10-31, after docs/hint-unselected.md): the legacy cohort (Qwen and DSv4 Flash
# consistent and disagreeing, Gemini 3.8 Flash at effort low, reads at the option newlines and the end of the prompt, draws
# averaged per case) beside the unselected one (consistency the only filter, Haiku 5.5 at medium, canonical reads, the
# `prompt` menu, draws pooled); hint first in both
PANELS = {
    "selected": ("DSv4-disagreement cohort", "Gemini Flash, legacy reads"),
    "selected_haiku": ("DSv4-disagreement cohort", "Haiku, legacy reads"),
    "unselected": ("unselected cohort", "Haiku, canonical reads"),
}


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
            ax.text(0, 1.2, PANELS[k][0], transform=ax.transAxes, fontsize=8.5 * SLIDE_S, fontweight="semibold", color=INK)
            ax.text(0, 1.08, f"{n:,} cases · {PANELS[k][1]}", transform=ax.transAxes, fontsize=6.5 * SLIDE_S, color=MUTED)
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
    """The two questions as rows (did Qwen know it? as AUROC, which letter? as % correct), hint first; columns: the legacy cohort
    as the talk had it (Flash), the legacy cohort rerun with Haiku (hint_unselected.LEGACY_HAIKU), the unselected cohort."""
    knew = {"selected": "hint_qwen_dsv4_before_tail_inline", "selected_haiku": "knew:legacy", "unselected": "knew"}
    letter = {"selected": "hint_qwen_dsv4_before_letter_tail_inline", "selected_haiku": "letter:legacy", "unselected": "letter"}

    def bars(legacy_bars):
        def f(name: str):
            task, *legacy = name.split(":")
            if task in ("knew", "letter"):
                return hint_unselected.talk_bars(task, hint_unselected.LEGACY_HAIKU[task] if legacy else None)
            return legacy_bars(name)

        return f

    knew_bars, letter_bars = bars(auroc_bars), bars(hinted_baseline(answer_bars))
    widths = [positions(knew_bars(n)[0], GROUPS)[1][-1] + 1 for n in knew.values()]
    fig, axes = slide_figure(FULL_W, 60, nrows=2, ncols=3, sharey="row", gridspec_kw={"width_ratios": widths})
    row(axes[0], knew, knew_bars, "chance", "did Qwen know it?\nAUROC of P(knew)", pct=False, head=True, base=0.5)
    row(axes[1], letter, letter_bars, "always the hinted letter", "which letter?\n% correct", pct=True, head=False, base=0.4)
    axes[0][0].text(1.0, 1.2, "hatched = placebo", transform=axes[0][0].transAxes, fontsize=6.5 * SLIDE_S, color=MUTED, ha="right")
    departure(fig, GRID_DEPARTURE)
    # the write-up's copy (docs/final-writeup.md, hinted multiple choice): the same figure on the page background
    fig.savefig(Path("docs/img/hinted-mcq-grid.png"), dpi=300, transparent=False, facecolor=BG)
    slide_save(fig, OUT / "grid.pdf")


@slide
def essay() -> None:
    """AUROC of P(the score goes up without the opener), graphs.essay_grading's chart at the deck's size (full slide width, as
    Part 1's figures), without its "+ mean" bars and the bet-at-the-mean line (Kylie 2026-11-01)."""
    fig, ax = slide_figure(FULL_W, 24)
    essay_bars(ax, SLIDE_S, ylabel="AUROC of P(up)", mean=False, null_line=False)  # the frame title asks "does Qwen's grade go up"
    ax.set_xticklabels([t.get_text() + "†" * (t.get_text() == "probe") for t in ax.get_xticklabels()])  # the footnote's dagger
    departure(fig, ESSAY_DEPARTURE)
    slide_save(fig, OUT / "essay.pdf")
    essay_tables()


@slide
def essay_diffs() -> None:
    """The paired differences as a forest plot in diff_rows' style (ink CI line and dot, filled when the CI excludes 0, a line
    at 0), compact for the slide's side column: the table above it carries the numbers and swatches. The same essay
    resamples as the table and the stars; the probe is left out so it doesn't set the axis."""
    pairs = essay_auroc_bars()[3]
    short = {"olens_inline": OLENS, "nla_inline": "NLA", "jlens_inline": "J++"}
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
    O-lens-vs-placebo flip counts (fig/essay_flips.tex) and the bet-at-the-mean AUROC the probe's dagger cites
    (fig/essay_null.tex) for the footnote."""
    pairs = essay_auroc_bars()[3]
    name = {"transcript_inline": r"\sw{transcript}\ transcript", "olens_inline": r"\sw{olens}\ \olensname",
            "olens_placebo_inline": r"\swp{olens}\ placebo", "nla_inline": r"\sw{nla}\ NLA", "nla_placebo_inline": r"\swp{nla}\ placebo",
            "jlens_inline": r"\sw{jlens}\ J++", "jlens_placebo_inline": r"\swp{jlens}\ placebo",
            "probe_user_end": r"\sw{probe}\ probe"}  # fmt: skip
    pf = lambda p: "< .001" if p < 1e-3 else f"{p:.3f}".lstrip("0")
    cell = lambda k: f"{pairs[k][0]:+.3f} & {pf(pairs[k][2])}" if k in pairs else "&"
    rows = [f"{name[a]} − {name[b].split(' ', 1)[1]} & {cell((a, b))}\\\\"
            for a, b in pairs if not a.endswith("+mean") and not a.startswith("probe")]  # fmt: skip
    tab = lambda spec, lines: f"\\begin{{tabular}}{{@{{}}{spec}@{{}}}}\n" + "\n".join(lines) + "\n\\end{tabular}\n"
    head = [r" & $\Delta$ & p\\"]
    (OUT / "essay_pairs.tex").write_text(tab("lrr", [*head, "\\hline", *rows]))
    (OUT / "essay_flips.tex").write_text("{} and {}".format(*essay_flips()))
    (OUT / "essay_null.tex").write_text(f"{essay_auroc_bars()[1]:.3f}")


@slide
def grader() -> None:
    """Two pairs of bars at the "." ending "If you are not sure, give your best guess.", chat template vs the submit_answer tool
    template, on the 863 cutoff_qa items read in both: the share of O-lens reads (six layers) saying "grader"/"graded" (not in the
    prompt), and the share of items with at least one such read, in a lighter shade of the bar's colour (hatching means placebo)."""
    run = Path(CUTOFF_GRADER_MENTIONS_RUN)
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


HARNESS_LABELS = {("system", "agent"): "agent\nprompt", ("system", "asst"): "asst.\nprompt", ("envblock", "env"): "env\nblock",
                  ("envblock", "noenv"): "no env\nblock", ("done", "done"): "done\ntool", ("done", "nodone"): "no done\ntool",
                  ("names", "plain"): "plain\nnames", ("names", "mcp"): "MCP\nnames", ("date", "now"): "date\nnow",
                  ("date", "past"): "date\npast"}  # fmt: skip


@slide
def acting_bars() -> None:
    """harness_grid thinking off, 2x2 (Kylie, 2026-11-01): rows Qwen3.6-27B and GLM-5.3 (one colour each, outside the condition
    palette), columns by wording and by each harness factor's two levels; per group, % acted without asking (all episodes) and,
    beside it in a lighter shade, % of the randomly sampled episodes with an O-lens read past the prompt saying
    grader/grading/graded. Wilson 95% CIs, shared y."""
    res = acting_by_factor()
    colours = {"Qwen3.6-27B": "#c2185b", "GLM-5.3": "#8c6d1f"}
    fig, axes = slide_figure(FULL_W, 50, nrows=2, ncols=2, sharey=True, gridspec_kw={"width_ratios": [9, 11.5]})
    for row, (model, r) in zip(axes, res.items()):
        light = to_hex(0.45 * np.array(to_rgb(colours[model])) + 0.55 * np.array(to_rgb(BG)))
        k_all, n_all = r["acted"]["all"]
        for ax, key in zip(row, ("wording", "harness")):
            xs = []
            for j, g in enumerate(r["acted"][key]):
                x = j + (j // 2) * 0.5 if key == "harness" else j  # gap between factors
                for off, what, col in [(-0.19, "acted", colours[model]), (0.19, "reads", light)]:
                    k, n = r[what][key][g]
                    lo, hi = wilson(k, n)
                    ax.bar(x + off, k / n, 0.38, color=col)
                    ax.errorbar(x + off, k / n, [[k / n - lo], [hi - k / n]], color=MUTED, lw=0.6, capsize=1.2, capthick=0.6)
                xs.append(x)
            ax.axhline(k_all / n_all, color=FAINT, lw=0.7, ls=":")
            labels = [w.replace("_", " ") for w in r["acted"][key]] if key == "wording" else [HARNESS_LABELS[f] for f in r["acted"][key]]
            ax.set_xticks(xs, labels, rotation=30 if key == "wording" else 0, ha="right" if key == "wording" else "center")
            ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
            ax.grid(axis="x", visible=False)
            ax.set_title(f"{model} thinking off, by {key}", loc="left", fontsize=7 * SLIDE_S, color=INK)
        row[0].set_ylabel("% of episodes")
        handles = [Rectangle((0, 0), 1, 1, color=c) for c in (colours[model], light)]
        row[1].legend(handles, ["acted without asking", "O-lens read says grader/grading/graded (past the prompt)"], loc="upper right",
                      frameon=False, ncol=2, fontsize=6 * SLIDE_S, bbox_to_anchor=(1, 1.22))  # fmt: skip
    axes[0][0].set_ylim(0, 0.82)
    axes[0][0].set_yticks(np.arange(0, 0.81, 0.2), [f"{t:.0%}" for t in np.arange(0, 0.81, 0.2)])
    slide_save(fig, OUT / "acting.pdf")
    acted = [f"{m} {r['acted']['all'][0] / r['acted']['all'][1]:.0%} of {r['acted']['all'][1]:,}" for m, r in res.items()]
    reads = [f"{m} {r['reads']['all'][1]:,}" for m, r in res.items()]
    tex = "\\def\\actAll{" + "; ".join(acted) + "}\n\\def\\readN{" + "; ".join(reads) + "}\n"
    (OUT / "acting.tex").write_text(tex.replace("%", "\\%"))


MANDARIN_LENSES = {"old": f"Old Qwen {OLENS}\n(spaces stripped)", "new": f"Qwen {OLENS}\n(fixed)"}


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
        ax.set_xticks(ticks, [OLENS if g == "olens" else g for g, _ in AUTHOR_GROUPS])
        ax.set_xlim(-0.6, width - 0.6)
        ax.tick_params(axis="x", labelsize=7 * SLIDE_S)
        ax.set_title(title, fontsize=7 * SLIDE_S, color=INK, loc="left")
        ax.grid(axis="x", visible=False)
    yt = np.arange(base, top, 0.1)
    axes[0].set_ylim(base, top)
    axes[0].set_yticks(yt, [f"{t:.1f}" for t in yt])
    axes[0].set_ylabel("AUROC of P(significant)")
    axes[-1].text(1.0, 1.12, "hatched = placebo", transform=axes[-1].transAxes, fontsize=6.5 * SLIDE_S, color=MUTED, ha="right")
    departure(fig, author_blog_drop.DEPARTURE)
    slide_save(fig, OUT / "author.pdf")
    name = {"transcript_inline": r"\sw{transcript}\ transcript", "olens_inline": r"\sw{olens}\ \olensname",
            "olens_placebo_inline": r"\swp{olens}\ \olensname{} placebo", "nla_inline": r"\sw{nla}\ NLA",
            "nla_placebo_inline": r"\swp{nla}\ NLA placebo", "transcript+probe": r"\sw{probe}\ probe + transcript"}  # fmt: skip
    pf = lambda p: "< .001" if p < 1e-3 else f"{p:.3f}".lstrip("0")
    rows = [
        f"{name[a]} − {name[c].split(' ', 1)[1]} & {d:+.3f} & [{lo:+.3f}, {hi:+.3f}] & {pf(p)}\\\\"
        for (a, c), (d, (lo, hi), p) in pairs.items()
    ]
    head = [r" & $\Delta$ AUROC & 95\% CI & p\\"]
    (OUT / "author_pairs.tex").write_text("\\begin{tabular}{@{}lrrr@{}}\n" + "\n".join([*head, "\\hline", *rows]) + "\n\\end{tabular}\n")
    (OUT / "author_n.tex").write_text(f"{n:,}")


@slide
def sankeys() -> None:
    """The write-up's data-filter Sankeys, one per dataset (fig/sankey_<slug>.pdf), the frame title carrying the title; and the
    two Multiple Choice sets stacked on one full-height figure (fig/sankey_mcq.pdf; Robbie, 2026-10-31)."""
    for slug, (_, columns, flows) in data_filter_sankey.DATASETS.items():
        fig, ax = slide_figure(FULL_W, 44)
        data_filter_sankey.draw(ax, columns, flows, fontsize=7.5 * SLIDE_S)
        slide_save(fig, OUT / f"sankey_{slug}.pdf")
    fig, axes = slide_figure(FULL_W, 58, nrows=2)
    for ax, slug in zip(axes, ("mmlu", "supergpqa"), strict=True):
        _, columns, flows = data_filter_sankey.DATASETS[slug]
        data_filter_sankey.draw(ax, columns, flows, fontsize=7.5 * SLIDE_S)
    slide_save(fig, OUT / "sankey_mcq.pdf")


# each task's results figure: the seven bars beside the three lens − transcript differences (Robbie, 2026-10-31), full
# width; Knowledge Cutoff's is shorter, its slide keeps three bullets under it
TASK_H = {"mcq": 54, "cutoff": 40, "preferences": 54}


@slide
def headline() -> None:
    """The write-up's headline at slide size: pooled over the three tasks, transcript only and the three lenses, then the lenses
    beside their placebos (fig/headline.pdf); the lenses against the probe on multiple choice as % right and as error rate
    (fig/headline_probes.pdf); and each task's seven bars with the paired lens − transcript differences beside them
    (fig/task_<task>.pdf), the axis from 50 as in the write-up."""
    for _, runs in MENUS.values():  # CI has no data/: fetch the canonical menus_inline runs it reads
        for run in runs:
            for f in ("meta.json", "predictions.jsonl"):
                if not (CF / run / f).exists():
                    hf_hub_download(HF_REPO, f"runs/cfbench/{run}/{f}", repo_type="dataset", local_dir="data")
    boot = Boot(headline_inline.load())
    lens_groups = [(label(c), [c]) for c in LENSES]
    pair_groups = [(label(c), [c, PLACEBOS[c]]) for c in LENSES]
    seven = [("prompt\nonly", [TS]), *pair_groups]
    probe_groups = [*lens_groups, ("probe", [PROBE])]
    fig, axes = slide_figure(FULL_W, 32, ncols=2, sharey=True, gridspec_kw={"width_ratios": [4, 6]})
    headline_inline.bars(axes[0], boot, POOLED, [("prompt\nonly", [TS]), *lens_groups], s=SLIDE_S)
    headline_inline.bars(axes[1], boot, POOLED, pair_groups, s=SLIDE_S)
    for ax, title in zip(axes, ("pooled over the three tasks", "each lens beside its placebo"), strict=True):
        ax.set_title(title, loc="left")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("% of cases right")
    slide_save(fig, OUT / "headline.pdf")
    fig, axes = slide_figure(FULL_W, 32, ncols=2)
    headline_inline.bars(axes[0], boot, ["mcq"], probe_groups, s=SLIDE_S)
    headline_inline.bars(axes[1], boot, ["mcq"], probe_groups, error=True, s=SLIDE_S)
    axes[0].set_title("Multiple Choice, % right", loc="left")
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("% of cases right")
    axes[1].set_title("Multiple Choice, error rate", loc="left")
    axes[1].set_ylim(0, 30)
    axes[1].set_ylabel("% of cases wrong")
    slide_save(fig, OUT / "headline_probes.pdf")
    for task, h in TASK_H.items():
        fig, (ax, dx) = slide_figure(FULL_W, h, ncols=2, gridspec_kw={"width_ratios": [3, 2]})
        headline_inline.bars(ax, boot, [task], seven, s=SLIDE_S)
        ax.set_ylim(50, 100)
        ax.set_ylabel("% of cases right")
        ax.text(1.0, 1.03, "hatched = placebo", transform=ax.transAxes, fontsize=6.5 * SLIDE_S, color=MUTED, ha="right")
        rows = [(c, TS, *boot.diff_ci([task], c, TS), label(c)) for c in LENSES]  # the swatches say "− transcript only"
        diff_rows(dx, rows, unit=" pp", s=SLIDE_S)
        lo, hi = min(0, *(r[3] for r in rows)), max(r[4] for r in rows)
        dx.set_xlim(lo - 0.08 * (hi - lo), hi + 1.1 * (hi - lo))  # room for the numbers right of the longest CI
        dx.set_xlabel("Δ pp vs transcript only, paired 95% CI")
        slide_save(fig, OUT / f"task_{task}.pdf")


@slide
def cutoff_investigators() -> None:
    """Knowledge cutoff by investigator (graphs.cutoff_qa_investigators, canonical set) at slide size: stale, not stale and all
    items as rows, the legend in the top row."""
    runs = {run for bars in cutoff_qa_investigators.SETS["canonical"].values() for _, rs, _ in bars for run in rs}
    need = [*cutoff_qa_investigators.CENSUS.values()]
    need += [run / f for run in runs for f in ("meta.json", "predictions.jsonl")]
    for path in need:  # CI has no data/: fetch what load() reads
        if not path.exists():
            hf_hub_download(HF_REPO, str(path.relative_to("data")), repo_type="dataset", local_dir="data")
    data = cutoff_qa_investigators.load("canonical")
    fig, axes = slide_figure(FULL_W, 50, nrows=3, sharex=True)
    handles = cutoff_qa_investigators.draw(axes, data, s=SLIDE_S)
    for ax in axes:
        ax.set_ylabel(None)
        ax.set_yticks([0, 50, 100])
    fig.supylabel("% naming Qwen's answer", fontsize=8 * SLIDE_S)
    axes[-1].legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=5, fontsize=7 * SLIDE_S)
    departure(fig, cutoff_qa_investigators.DEPARTURE)
    slide_save(fig, OUT / "cutoff_investigators.pdf")


DRINK_ARMS = {"one": "1", "two": "2", "couple": "a couple\n(original)", "three": "3", "four": "4", "five": "5"}  # up to 5 (Kylie)


@slide
def drink_drive() -> None:
    """drink_drive: share of Qwen's replies the Haiku judge labels drive or mixed (agrees, or agrees with a condition), per drink
    count, Wilson 95% CIs. One series, so ink."""
    path = Path("data") / DRINK_RUN / "labels/judge@haiku55-v1.jsonl"
    if not path.exists():
        hf_hub_download(HF_REPO, f"{DRINK_RUN}/labels/judge@haiku55-v1.jsonl", repo_type="dataset", local_dir="data")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    fig, ax = slide_figure(62, 52)
    for x, arm in enumerate(DRINK_ARMS):
        ys = [r["values"]["drive_or_mixed"] for r in rows if r["episode"].split("~")[0] == arm]
        v, (lo, hi) = sum(ys) / len(ys), wilson(sum(ys), len(ys))
        ax.bar(x, v, 0.8, color=INK)
        ax.errorbar(x, v, [[v - lo], [hi - v]], color=MUTED, lw=0.9, capsize=3, capthick=0.9)
        ax.annotate(f"{v:.0%}", (x, hi), xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom",
                    fontsize=6.5 * SLIDE_S, color=INK)  # fmt: skip
    ax.set_xticks(range(len(DRINK_ARMS)), list(DRINK_ARMS.values()))
    ax.tick_params(axis="x", labelsize=7 * SLIDE_S)
    ax.set_xlabel("drinks the user says they had")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.25), [f"{t:.0%}" for t in np.arange(0, 1.01, 0.25)])
    ax.set_ylabel("tells the user to drive")
    ax.grid(axis="x", visible=False)
    slide_save(fig, OUT / "drink_drive.pdf")


# The Great Sneeze anecdote: CHIVE's Qwen3.5-397B incident prompt_003654 (adamkarvonen/chive-data), the rate of a sneeze-based
# event in Vardania's history over 30 samples, the original conversation and each of CHIVE's edits; the edit labels paraphrase
# Adam's experiment summaries. Not a condition, so a colour outside the conditions'.
SNEEZE_EDITS = {
    1: "coastal tropical, not landlocked",
    2: "formerly in the Serbanian Empire",
    3: "one turn, country + “make it absurd…”",
    4: "fun fact: spice bazaars, not mountains",
    5: "coastal + formerly in an empire",
    6: "Landlocked: No, mountains kept",
    7: "renamed Tuvalia, new culture",
    8: "fun fact: landmarks, not mountains",
    9: "independence + colony lines removed",
    10: "landlocked, fun fact about beaches",
    11: "one turn, no “absurd and funny and wacky”",
    12: "“absurd and funny and wacky” removed",
}
SNEEZE_C = "#5b6c8f"


@slide
def sneeze() -> None:
    """Horizontal bars: share of Qwen3.5-397B's 30 samples with a sneeze-based event in Vardania's history (CHIVE's grader), the
    original conversation on top in ink, then each edit by rate, with 95% Wilson whiskers."""
    path = Path("data") / SNEEZE_RUN
    if not path.exists():
        hf_hub_download(HF_REPO, str(SNEEZE_RUN), repo_type="dataset", local_dir="data")
    r = next(r for r in map(json.loads, path.open()) if r["prompt_id"] == "prompt_003654")
    edits = sorted(((SNEEZE_EDITS[e["exp"]], e["k"], e["n"]) for e in r["exps"]), key=lambda x: -x[1] / x[2])
    rows = [("original conversation", r["k0"], r["n0"]), *edits]
    fig, ax = slide_figure(64, 44)
    for y, (_, k, n) in enumerate(rows):
        lo, hi = wilson(k, n)
        ax.barh(y, k / n, 0.7, color=INK if y == 0 else SNEEZE_C, zorder=3)
        ax.errorbar(k / n, y, xerr=[[k / n - lo], [hi - k / n]], color=MUTED, lw=0.9, capsize=2, capthick=0.9, zorder=4)
        ax.annotate(f"{k}/{n}", (hi, y), xytext=(2, 0), textcoords="offset points", va="center", fontsize=6.5 * SLIDE_S, color=INK)
    ax.set_yticks(range(len(rows)), [x[0] for x in rows])
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 0.88)
    ax.set_xticks(np.arange(0, 0.76, 0.25), [f"{t:.0%}" for t in np.arange(0, 0.76, 0.25)])
    ax.set_xlabel("samples with a sneeze-based event")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    slide_save(fig, OUT / "sneeze.pdf")


if __name__ == "__main__":
    grid()
    essay()
    essay_diffs()
    grader()
    acting_bars()
    mandarin()
    author()
    sankeys()
    headline()
    cutoff_investigators()
    drink_drive()
    sneeze()
