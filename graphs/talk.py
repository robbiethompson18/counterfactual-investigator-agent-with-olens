"""The sprint talk's figures (slides/talk/talk.tex), PDFs at their slide size in slides/talk/fig/, plus the LaTeX table
snippets the deck \\inputs. Statistics come from cfbench.scoring (the hint evals' bars from the viewer's evals/NAME.json, essay
grading's and the blog deletions' AUROCs from the runs' predictions; the essay bars themselves are graphs.essay_grading's, drawn
here at slide size), the look from graphs.style (@slide: the deck's sizes and a
transparent background around each figure). The write-up's dataset parts (docs/final-writeup.md Part 1) are drawn here at slide
size from their write-up modules: the Sankeys (graphs.data_filter_sankey), the headline and per-task bars (graphs.headline_inline)
and the knowledge-cutoff investigators (graphs.cutoff_qa_investigators). `make -C slides` runs this first.

    uv run python -m graphs.talk   # -> slides/talk/fig/*.pdf, *.tex
"""

import ast
import json
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter

from cfbench.catalog import CF, CHIVE_AGAM_RUNS, CHIVE_RUN, CUTOFF_GRADER_MENTIONS_RUN, DRINK_RUN, MANDARIN_RUN, MENUS
from cfbench.scoring import CONDS as CF_CONDS
from cfbench.scoring import (
    ESSAY_LENSES,
    HF_REPO,
    PROMPT_SECTIONS,
    acting_by_factor,
    answer_bars,
    auroc_bars,
    essay_auroc_bars,
    essay_flips,
    simulated_by_section,
)
from graphs import (
    author_blog_drop,
    chive_agam_inline,
    chive_inline,
    cutoff_qa_investigators,
    data_filter_sankey,
    great_sneeze,
    headline_ablations,
    headline_inline,
    headline_tokens,
    menus_inline,
    read_positions_diagram,
    wsb_text_inversion,
)
from graphs.cfbench_modelchart import DUMB_MODELS, MODEL_AXIS, MODELS, THINKING, THINKING_ON, case_ci
from graphs.cfbench_report import DEPARTURE as CF_DEPARTURE
from graphs.cfbench_report import TOOL_CONDS, TOOLS
from graphs.cutoff_grader_mentions import RUNS, at_token, wilson
from graphs.essay_grading import DEPARTURE as ESSAY_DEPARTURE
from graphs.essay_grading import essay_bars
from graphs.headline_inline import LENSES, PLACEBOS, POOLED, PROBE, TITLES, TS, Boot
from graphs.olens_mandarin import HAN, bullets_of
from graphs.style import (
    BG,
    FAINT,
    FULL_W,
    HALF_W,
    INK,
    MM,
    MUTED,
    OLENS,
    SLIDE_S,
    bar_kw,
    color,
    departure,
    diff_rows,
    label,
    positions,
    slide,
    slide_figure,
    slide_save,
)

GRID_DEPARTURE = (f"investigator Gemini 3.8 Flash at effort low; {OLENS} at layers 32/40/48/52; positions are the option newlines and "
                  "the end of the prompt only.")  # fmt: skip

OUT = Path("slides/talk/fig")
# the page's order in groups, each lens with its hatched placebo beside it under one label (Kylie, 2026-10-30); O-lens
# (nl) is left out (Kylie, 2026-10-31).
GROUPS = [("transcript", ["transcript_inline"]), (OLENS, ["olens_inline", "olens_placebo_inline"]),
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
    fig, ax = slide_figure(72, 20)
    ticks, labels, y = [], [], 0.0
    for a in ESSAY_LENSES:  # each lens vs transcript only; no lens − placebo rows (Robbie, 2026-10-31)
        d, (lo, hi), _ = pairs[a, "transcript_inline"]
        ax.plot([lo, hi], [y, y], color=INK, lw=1.2, solid_capstyle="butt", zorder=3)
        ax.plot(d, y, "o", ms=3.5, zorder=4, mec=INK, mfc=INK if lo > 0 or hi < 0 else BG)
        ticks.append(y)
        labels.append(f"{short[a]} − transcript")
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
            "nla_inline": r"\sw{nla}\ NLA", "jlens_inline": r"\sw{jlens}\ J++"}  # fmt: skip
    pf = lambda p: "< .001" if p < 1e-3 else f"{p:.3f}".lstrip("0")
    cell = lambda k: f"{pairs[k][0]:+.3f} & {pf(pairs[k][2])}" if k in pairs else "&"
    rows = [f"{name[a]} − {name[b].split(' ', 1)[1]} & {cell((a, b))}\\\\"
            for a, b in pairs if b == "transcript_inline" and a in ESSAY_LENSES]  # fmt: skip
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


@slide
def simulated_sections() -> None:
    """harness_grid prompt reads: % of O-lens draws saying simulated/mock/synthetic/fictional by prompt section,
    Qwen3.6-27B beside GLM-5.3 (the acting slide's model colours), Wilson 95% CIs (draws within a prompt aren't independent)."""
    res = simulated_by_section()
    colours = {"Qwen3.6-27B": "#c2185b", "GLM-5.3": "#8c6d1f"}
    fig, ax = slide_figure(HALF_W, 48)
    for i, (model, r) in enumerate(res.items()):
        for j, sec in enumerate(PROMPT_SECTIONS):
            k, n = r[sec]
            lo, hi = wilson(k, n)
            x = j + (i - 0.5) * 0.38
            ax.bar(x, k / n, 0.36, color=colours[model], label=model if j == 0 else None)
            ax.errorbar(x, k / n, [[k / n - lo], [hi - k / n]], color=MUTED, lw=0.6, capsize=1.2, capthick=0.6)
            ax.annotate(f"{100 * k / n:.1f}", (x, hi), xytext=(0, 1), textcoords="offset points", ha="center", va="bottom",
                        fontsize=5.5 * SLIDE_S, color=INK)  # fmt: skip
    ax.set_xticks(range(len(PROMPT_SECTIONS)), PROMPT_SECTIONS)
    ax.set_ylim(0, 0.27)
    ax.set_yticks(np.arange(0, 0.26, 0.05), [f"{t:.0%}" for t in np.arange(0, 0.26, 0.05)])
    ax.set_ylabel("reads saying simulated, mock, ...")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left")
    slide_save(fig, OUT / "simulated_sections.pdf")


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
    """The blog-author results slide: graphs.author_blog_drop.slide()."""
    author_blog_drop.slide(OUT)


@slide
def sankeys() -> None:
    """The write-up's data-filter Sankeys, one per dataset (fig/sankey_<slug>.pdf), the frame title carrying the title; and the
    three single-answer tasks' Sankeys at a column's width, beside their part's bullets (Robbie, 2026-10-31): Knowledge Cutoff
    and Idiosyncratic Preferences as fig/sankey_<slug>_half.pdf, the two Multiple Choice sets stacked on one wider figure
    (fig/sankey_mcq.pdf)."""
    for slug, (_, columns, flows) in data_filter_sankey.DATASETS.items():
        fig, ax = slide_figure(FULL_W, 44)
        data_filter_sankey.draw(ax, columns, flows, fontsize=7.5 * SLIDE_S)
        slide_save(fig, OUT / f"sankey_{slug}.pdf")
    for slug in ("cutoff", "favorites"):
        _, columns, flows = data_filter_sankey.DATASETS[slug]
        fig, ax = slide_figure(HALF_W, 50)
        data_filter_sankey.draw(ax, columns, flows, fontsize=6.5 * SLIDE_S)
        slide_save(fig, OUT / f"sankey_{slug}_half.pdf")
    fig, axes = slide_figure(90, 56, nrows=2)
    for ax, slug in zip(axes, ("mmlu", "supergpqa"), strict=True):
        _, columns, flows = data_filter_sankey.DATASETS[slug]
        data_filter_sankey.draw(ax, columns, flows, fontsize=6 * SLIDE_S)
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
    fig, axes = slide_figure(FULL_W, 55, nrows=3, sharex=True)
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


@slide
def sneeze() -> None:
    """Horizontal bars: share of Qwen3.5-397B's 30 samples with a sneeze-based event in Vardania's history (CHIVE's grader), the
    original conversation on top in ink, then each edit by rate, with 95% Wilson whiskers."""
    rows = great_sneeze.rows()
    fig, ax = slide_figure(64, 44)
    for y, (_, k, n) in enumerate(rows):
        lo, hi = wilson(k, n)
        ax.barh(y, k / n, 0.7, color=INK if y == 0 else great_sneeze.SNEEZE_C, zorder=3)
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


# --- the counterfactual part (the write-up's Part I and its appendix of counterfactual charts) --------------------------------


@slide
def chive() -> None:
    """CHIVE's own yes/no claims: Haiku edit only vs transcript beside CHIVE's Opus 4.8 arms, one panel per target and pooled
    (fig/chive.pdf), and the paired transcript − edit-only rows (fig/chive_diff.pdf), graphs.chive_inline at slide size."""
    panels, rows = chive_inline.panels_data(CHIVE_RUN)
    fig, axes = slide_figure(FULL_W, 34, ncols=len(panels), sharey=True)
    chive_inline.draw(axes, panels, SLIDE_S)
    slide_save(fig, OUT / "chive.pdf")
    fig, ax = slide_figure(HALF_W, 18)
    diff_rows(ax, rows, value_fmt="{:+.3f}", s=SLIDE_S)
    ax.set_xlim(-0.02, 0.05)
    ax.set_xlabel("Haiku AUROC, transcript − edit only, paired by claim (95% CI)")
    slide_save(fig, OUT / "chive_diff.pdf")


@slide
def chive_agam() -> None:
    """CHIVE-style claims on Agam's Qwen3.6-27B incidents: edit only, transcript, each lens beside its placebo
    (fig/chive_agam.pdf; the claim count in fig/chive_agam_n.tex), graphs.chive_agam_inline at slide size."""
    fig, ax = slide_figure(HALF_W, 46)
    n, _ = chive_agam_inline.bars(ax, CHIVE_AGAM_RUNS, SLIDE_S)
    ax.text(1.0, 1.02, "hatched = placebo", transform=ax.transAxes, fontsize=6.5 * SLIDE_S, color=MUTED, ha="right")
    departure(fig, chive_agam_inline.DEPARTURE)
    slide_save(fig, OUT / "chive_agam.pdf")
    (OUT / "chive_agam_n.tex").write_text(f"{n:,}")


CF_SUMMARY = Path("docs/img/cfbench-report/summary.json")  # every number on graphs.cfbench_report's charts


def cf_acc(rmse: float) -> float:
    """What the counterfactual bars show: 100 − RMSE in percentage points, higher is better (docs/cfbench-scoring.md)."""
    return 100 - 100 * rmse


def cf_bars(ax, s: dict, x0: float, width: float, legend: bool) -> None:
    """One stats dict's five bars (the four conditions and the edit-does-nothing null) with their bootstrap intervals, as
    graphs.cfbench_report.bar_group draws them, at the deck's sizes."""
    series = [(c, s["rmse"][c], s["rmse_ci"][c]) for c in CF_CONDS] + [("null", s["rmse_null"], s["rmse_null_ci"])]
    for j, (c, v, (lo, hi)) in enumerate(series):
        x = x0 + (j - 2) * width
        ax.bar(x, cf_acc(v), width * 0.9, label=label(c) if legend else None, zorder=2, **bar_kw(c))
        ax.plot([x, x], [cf_acc(hi), cf_acc(lo)], color=INK, lw=0.8, zorder=3)
        ax.text(x, cf_acc(lo) + 0.6, f"{cf_acc(v):.1f}", ha="center", va="bottom", fontsize=6.5 * SLIDE_S, color=INK)


@slide
def cfbench() -> None:
    """The counterfactual charts, from graphs.cfbench_report's summary.json (run `python -m graphs.cfbench_report` first, ~20 s):
    100 − RMSE pooled and per target (fig/cf_skill.pdf), the pre-registered hidden_goal.grade test (fig/cf_hidden_goal.pdf), what
    the O-lens agent searches the lens for and which tools it calls (fig/cf_harness.pdf), investigator model by condition and
    transcript-only vs O-lens per investigator (fig/cf_models.pdf), Gemini 3.8 Flash by effort and by tool budget
    (fig/cf_flash.pdf)."""
    assert CF_SUMMARY.exists(), f"{CF_SUMMARY} missing: run `uv run python -m graphs.cfbench_report` first"
    S = json.loads(CF_SUMMARY.read_text())
    # pooled and by target
    fig, (ax, bx) = slide_figure(FULL_W, 46, ncols=2, sharey=True, gridspec_kw={"width_ratios": [1, 2.4]})
    cf_bars(ax, S["pooled"], 0, 0.16, legend=True)
    ax.axhline(cf_acc(S["pooled"]["noise_floor"]), color=MUTED, ls=":", lw=0.8, zorder=1, label="ceiling (sampling noise)")
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_ylim(0, 130)
    ax.set_ylabel("100 − RMSE of predicted rate (pp)")
    ax.set_title(f"all targets, {S['pooled']['cases']} cases, {S['pooled']['envs']} envs", loc="left")
    targets = list(S["by_target"])
    for i, t in enumerate(targets):
        st = S["by_target"][t]
        cf_bars(bx, st, i, 0.16, legend=False)
        nf = cf_acc(st["noise_floor"])
        bx.plot([i - 0.45, i + 0.45], [nf, nf], color=MUTED, ls=":", lw=0.8, zorder=1)
    bx.set_xticks(range(len(targets)), [f"{t}\n{S['by_target'][t]['cases']} cases, {S['by_target'][t]['envs']} envs" for t in targets])
    bx.set_title("by target model", loc="left")
    bx.legend(*ax.get_legend_handles_labels(), loc="upper left", ncol=3)
    for a in (ax, bx):
        a.grid(axis="x", visible=False)
    departure(fig, CF_DEPARTURE)
    slide_save(fig, OUT / "cf_skill.pdf")
    # hidden_goal.grade
    fig, ax = slide_figure(HALF_W, 40)
    hg = S["hidden_goal"]
    cf_bars(ax, hg, 0, 0.16, legend=True)
    ax.axhline(cf_acc(hg["noise_floor"]), color=MUTED, ls=":", lw=0.8, zorder=1, label="ceiling (sampling noise)")
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_ylim(0, 100)
    ax.set_ylabel("100 − RMSE of predicted rate (pp)")
    ax.set_title(f"hidden_goal.grade, Qwen3.6-27B without CoT, {hg['cases']} arm pairs", loc="left")
    fig.legend(loc="outside lower center", ncol=3)
    ax.grid(axis="x", visible=False)
    departure(fig, CF_DEPARTURE)
    slide_save(fig, OUT / "cf_hidden_goal.pdf")
    # harness sanity checks: search topics and tool calls
    fig, (ax, bx) = slide_figure(FULL_W, 42, ncols=2, gridspec_kw={"width_ratios": [1.15, 1]})
    T = S["topics"]
    cats = list(T["share"])  # most common first
    ax.barh(range(len(cats)), [T["share"][c] for c in cats], 0.6, color=color("olens_agentic"))
    for i, c in enumerate(cats):
        ax.text(T["share"][c] + 0.005, i, f"{T['share'][c]:.0%}", va="center", fontsize=7 * SLIDE_S)
        rx = next((r for r, _ in T["examples"][c] if len(r) <= 40), "")
        ax.text(-0.01, i + 0.38, f"e.g. /{rx}/", ha="right", va="center", fontsize=6 * SLIDE_S, color=MUTED,
                transform=ax.get_yaxis_transform())  # fmt: skip
    ax.set_yticks(range(len(cats)), cats)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel(f"share of {T['calls']:,} search_olens calls")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"what the {OLENS} agent searches the lens for", loc="left")
    tools, w = S["tools"], 0.8 / 3
    for j, c in enumerate(TOOL_CONDS):
        for i, tool in enumerate(TOOLS):
            v, x = tools[c][tool], i + (j - 1) * w
            bx.bar(x, v, w * 0.9, label=f"{label(c)} ({tools[c]['predictions']:,} predictions)" if i == 0 else None, **bar_kw(c))
            bx.text(x, v + 0.1, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5 * SLIDE_S)
    bx.set_xticks(range(len(TOOLS)), TOOLS)
    bx.set_ylabel("calls per prediction")
    bx.grid(axis="x", visible=False)
    bx.legend(loc="upper right")
    bx.set_title("which tools the investigator calls (20-call budget)", loc="left")
    slide_save(fig, OUT / "cf_harness.pdf")
    # investigator model by condition, and transcript-only vs O-lens per investigator
    R = S["robustness"]
    models = {ast.literal_eval(k): v for k, v in R["models"].items()}
    fig, (ax, bx) = slide_figure(FULL_W, 52, ncols=2, gridspec_kw={"width_ratios": [1.25, 1]})
    xs, ends = range(len(MODEL_AXIS)), []
    for m, name in MODELS.items():
        if any((m, c) not in models for c in MODEL_AXIS):
            continue
        ys = [models[m, c]["acc"] for c in MODEL_AXIS]
        ax.plot(xs, ys, color="#c8c8c8", lw=1, zorder=1)
        for x, c, y in zip(xs, MODEL_AXIS, ys, strict=True):
            ax.scatter(x, y, s=16, color=color(c), zorder=3)
        ends.append((ys[-1], name))
    label_y = -1e9
    for y, name in sorted(ends):  # end labels at least 1.6pp apart
        label_y = max(y, label_y + 1.6)
        ax.text(len(MODEL_AXIS) - 1 + 0.12, label_y, name, va="center", fontsize=7 * SLIDE_S)
    ax.set_xticks(list(xs), [label(c) for c in MODEL_AXIS])
    for tick, c in zip(ax.get_xticklabels(), MODEL_AXIS, strict=True):
        tick.set_color(color(c))
    ref = next(iter(models.values()))
    ax.axhline(ref["null"], color=INK, lw=0.8, ls="--", zorder=1)
    ax.text(-0.3, ref["null"], " edit does nothing", va="bottom", fontsize=7 * SLIDE_S, color=INK)
    ax.set_xlim(-0.3, len(MODEL_AXIS) + 0.7)
    ax.set_ylabel("100 − RMSE of predicted rate (pp)")
    ax.set_title("investigator model by condition (effort high, 20 tool calls, 4 cases)", loc="left")
    tv = {ast.literal_eval(k): v for k, v in R["ts_vs_olens"].items()}
    names = MODELS | DUMB_MODELS
    ts, ol = "transcript_agentic", "olens_agentic"
    pts = sorted((tv[m, e, ts]["acc"], tv[m, e, ol]["acc"], m, e) for m, e, c in tv if c == ol and (m, e, ts) in tv)
    lo, hi = 34, 62
    bx.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, zorder=1)  # the diagonal: O-lens = transcript only
    null = next(iter(tv.values()))["null"]
    bx.axvline(null, color=INK, lw=0.8, ls="--", zorder=1)
    bx.axhline(null, color=INK, lw=0.8, ls="--", zorder=1)
    bx.text(null + 0.3, lo + 0.3, "edit does nothing", fontsize=6.5 * SLIDE_S, color=INK, rotation=90, va="bottom")
    bunch = sorted((p for p in pts if p[0] < 46 and p[1] < 46), key=lambda p: p[1])  # the cheap investigators, near the null
    column = {(m, e): lo + 1.2 + 1.45 * k for k, (_, _, m, e) in enumerate(bunch)}  # their labels, a column with leader lines
    for x, y, m, e in pts:
        _, shade = THINKING.get(e, THINKING_ON)
        bx.scatter(x, y, s=14, color=shade, edgecolor=INK, lw=0.4, zorder=3)
        name = names[m] + (f" ({e})" if sum(p[2] == m for p in pts) > 1 else "")
        if (m, e) in column:
            bx.annotate(name, (x, y), xytext=(47.5, column[m, e]), va="center", fontsize=5.5 * SLIDE_S, color=INK,
                        arrowprops={"arrowstyle": "-", "color": "#b5b5b5", "lw": 0.4, "shrinkA": 0, "shrinkB": 2})  # fmt: skip
        else:
            bx.annotate(name, (x, y), xytext=(2.5, 0), textcoords="offset points", va="center", fontsize=5.5 * SLIDE_S, color=INK)
    for level, shade in [*dict(THINKING.values()).items(), THINKING_ON]:
        bx.scatter([], [], s=14, color=shade, edgecolor=INK, lw=0.4, label=f"thinking {level}")
    bx.legend(loc="upper left")
    bx.set_xlim(lo, hi)
    bx.set_ylim(lo, hi)
    bx.set_aspect("equal")
    bx.set_xlabel("transcript only: 100 − RMSE (pp)")
    bx.set_ylabel(f"with {OLENS}: 100 − RMSE (pp)")
    bx.set_title(f"transcript only vs {OLENS} by investigator (4 cases)", loc="left")
    slide_save(fig, OUT / "cf_models.pdf")
    # Flash by effort and by tool budget
    fig, axes = slide_figure(FULL_W, 40, ncols=2, sharey=True)
    sweeps = (("flash_effort", ["low", "medium", "high"], "reasoning effort (20 tool calls)"),
              ("flash_calls", None, "max tool calls (effort high)"))  # fmt: skip
    for ax, (key, steps, xlabel) in zip(axes, sweeps, strict=True):
        s = {ast.literal_eval(k): v for k, v in R[key].items()}
        steps = steps or sorted({k[0] for k in s if k[1] == ts})
        for c in ["no_transcript", ts, ol]:
            present = [(i, s[x, c]) for i, x in enumerate(steps) if (x, c) in s]
            ax.plot([i for i, _ in present], [g["acc"] for _, g in present], color=color(c), marker="o", ms=3, lw=1.5, label=label(c))
            cis = [case_ci(g["per_case"]) for _, g in present]
            ax.fill_between([i for i, _ in present], [lo for lo, _ in cis], [hi for _, hi in cis], color=color(c), alpha=0.12, lw=0)
        ref = next(iter(s.values()))
        ax.axhline(ref["null"], color=INK, lw=0.8, ls="--", zorder=1)
        ax.axhline(ref["ceiling"], color=MUTED, lw=0.8, ls=":", zorder=1)
        ax.set_xticks(range(len(steps)), [str(x) for x in steps])
        ax.set_xlabel(xlabel)
        ax.set_title(f"Gemini 3.8 Flash by {xlabel.split(' (')[0]}, 4 cases", loc="left")
    for y, text, col in ((ref["null"], "edit does nothing", INK), (ref["ceiling"], "ceiling (sampling noise)", MUTED)):
        axes[0].text(0.01, y, text, transform=axes[0].get_yaxis_transform(), va="bottom", fontsize=6.5 * SLIDE_S, color=col)
    axes[0].set_ylim(20, 100)
    axes[0].set_ylabel("100 − RMSE of predicted rate (pp)")
    axes[1].legend(loc="lower left")
    slide_save(fig, OUT / "cf_flash.pdf")


# --- the ablations part (the write-up's Ablations) ---------------------------------------------------------------------------


@cache
def canonical_boot() -> Boot:
    """The paired bootstrap over the canonical menus_inline runs, built once for every figure that pools the three tasks."""
    return Boot(headline_inline.load())


@slide
def correctness() -> None:
    """Prompt only and the three lenses on the cases where Qwen is right (top row) and wrong (bottom), MMLU-Pro, SuperGPQA and
    Knowledge Cutoff (fig/correctness.pdf), graphs.headline_ablations at slide size."""
    sets = headline_ablations.by_correctness()
    fig, axes = slide_figure(FULL_W, 46, nrows=2, ncols=3, sharey=True)
    for row, right in zip(axes, (True, False), strict=True):
        headline_ablations.accuracy_axes(row, sets, right, SLIDE_S)
        row[0].set_ylabel("% of cases right")
    slide_save(fig, OUT / "correctness.pdf")


@slide
def lens_diffs() -> None:
    """Each lens minus prompt only, paired by case, per task and pooled (fig/lens_diffs.pdf)."""
    boot = canonical_boot()
    rows: list = []
    for task in [*POOLED, "pooled"]:
        rows.append("pooled over the three tasks" if task == "pooled" else TITLES[task])
        rows += [(c, TS, *boot.diff_ci(POOLED if task == "pooled" else [task], c, TS)) for c in LENSES]
    fig, ax = slide_figure(FULL_W, 50)
    diff_rows(ax, rows, s=SLIDE_S)
    ax.set_xlim(-11, 12)
    ax.set_xlabel("difference in % of cases right, paired by case (95% bootstrap CI; filled = excludes 0)")
    slide_save(fig, OUT / "lens_diffs.pdf")


@slide
def read_positions() -> None:
    """One MMLU-Pro prompt with its question and tail read positions (fig/positions.pdf; no layout engine, so the token widths
    measured in axes units stay true) and the nine bars of each lens at the question, the tail and both (fig/menus.pdf)."""
    fig = plt.figure(figsize=(HALF_W * MM, 50 * MM))
    read_positions_diagram.draw(fig.add_axes((0, 0, 1, 1)), SLIDE_S)
    slide_save(fig, OUT / "positions.pdf")
    fig, ax = slide_figure(HALF_W, 50)
    menus_inline.menu_bars(ax, menus_inline.load(), SLIDE_S)
    slide_save(fig, OUT / "menus.pdf")


@slide
def placebos() -> None:
    """Prompt only and the three placebos, pooled over the three tasks (fig/placebos.pdf); Haiku's thinking tokens per
    condition (fig/thinking.pdf); prompt tokens each lens adds, pooled and per set (fig/tokens.pdf)."""
    fig, ax = slide_figure(HALF_W, 40)
    groups = [("prompt\nonly", [TS]), *((label(PLACEBOS[c]).replace(" placebo", "\nplacebo"), [PLACEBOS[c]]) for c in LENSES)]
    headline_inline.bars(ax, canonical_boot(), POOLED, groups, s=SLIDE_S)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of cases right")
    slide_save(fig, OUT / "placebos.pdf")
    sets = {name: headline_tokens.load(runs) for name, runs in headline_tokens.SETS.items()}
    fig, ax = slide_figure(HALF_W, 40)
    headline_tokens.thinking_ax(ax, headline_tokens.thinking_means(sets), SLIDE_S)
    slide_save(fig, OUT / "thinking.pdf")
    fig, ax = slide_figure(FULL_W, 40)
    headline_tokens.chart_ax(ax, {c: headline_tokens.summary(sets, c, "prompt") for c in LENSES}, SLIDE_S)
    slide_save(fig, OUT / "tokens.pdf")


@slide
def wsb() -> None:
    """WorkspaceBench's four contested families: the best published metamodel vs Sonnet 5.5 text inversion, at WSB's any-of-grid
    scoring and at pass@1 (fig/wsb.pdf), graphs.wsb_text_inversion at slide size."""
    fig, axes = slide_figure(FULL_W, 38, ncols=2, sharey=True)
    wsb_text_inversion.draw(axes[0], wsb_text_inversion.LENS_HEADLINE, f"{OLENS}, WSB headline (any readout of the grid passes)", SLIDE_S)
    wsb_text_inversion.draw(axes[1], wsb_text_inversion.LENS_PER_READOUT, f"{OLENS}, scored per readout (pass@1)", SLIDE_S)
    axes[0].set_title("WorkspaceBench's own scoring: any of 55 to ~230 readouts per item", loc="left")
    axes[1].set_title("both at pass@1: the lens scored per readout", loc="left")
    handles = [h for ax in axes for h in ax.get_legend_handles_labels()[0]]
    labels = [lab for ax in axes for lab in ax.get_legend_handles_labels()[1]]
    fig.legend(handles[:1] + handles[2:], labels[:1] + labels[2:], loc="outside lower center", ncol=3)
    slide_save(fig, OUT / "wsb.pdf")


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
    chive()
    chive_agam()
    cfbench()
    correctness()
    lens_diffs()
    read_positions()
    placebos()
    wsb()
    simulated_sections()
