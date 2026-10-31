"""The hint evals' bars, one panel per eval as on its Evals page (#eval=NAME), from the real results: the viewer's
evals/NAME.json (data/evals/, or HF's evals/ when it isn't local).

- Did Qwen know it? (hint_qwen_dsv4{,_before}_tail_inline, claim evals): % right with guess = p >= 0.5 and the 95% CI
  from claim_tables (cfbench/evals/<name>/score.py), stars from its paired one-sided sign test against transcript-only.
- Which letter? (hint_qwen_dsv4{,_before}_letter_tail_inline, answer evals): the page's headline answer panel recomputed
  as page.html's answerChartBlock does: every item's draws merged across runs, a bar is the mean over items of the item's
  fraction right, a 95% item bootstrap (2000 resamples; the page's own RNG differs, so a CI end can move by a point), the
  exact one-sided sign test against transcript-only over the items that differ, chance = 1 / the most options.

    uv run python -m slides.talk.figures   # -> slides/talk/fig/*.pdf
"""

import json
import math
import urllib.error
import urllib.request
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download
from matplotlib.colors import to_hex, to_rgb

from cfbench.evals.essay_grading_cf_inline.eval import mean
from cfbench.scoring import essay_lens_rows, essay_probe_scores
from graphs.cutoff_grader_mentions import RUNS, at_token
from graphs.style import BG, INK, MUTED, bar_kw, positions
from slides.figure import FULL_H, FULL_W, figure, save

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("fig")
HF_REPO = "Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens"
HF = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/evals/"
FAINT = "#a29d92"  # the Evals page's --faint
S = 2 / 3  # this deck's text is 2/3 of the theme's (Kylie, 2026-10-30), so the figures' is too
plt.rcParams.update({k: plt.rcParams[k] * S for k in ["font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize",
                                                      "ytick.labelsize", "legend.fontsize"]})  # fmt: skip
# the page's order in groups, each lens with its hatched placebo beside it under one label (Kylie, 2026-10-30). The
# O-lens (nl) placebo isn't on the tail pages: it is the placebo of the option-newlines-only eval the O-lens (nl) bar
# comes from (NAME without "_tail", same cases).
GROUPS = [("transcript", ["transcript_inline"]), ("O-lens", ["olens_inline", "olens_placebo_inline"]),
          ("O-lens (nl)", ["olens_inline_nl", "olens_nl_placebo"]), ("probe", ["probe"]), ("tuned\nprobe", ["probe_tuned"])]  # fmt: skip
PANELS = {"after": "hint after the options", "before": "hint first"}


def load(name: str) -> dict:
    local = ROOT / "data/evals" / f"{name}.json"
    if local.exists():
        return json.loads(local.read_text())
    with urllib.request.urlopen(HF + f"{name}.json") as r:
        return json.load(r)


def stars(p: float | None) -> str:
    return "" if p is None else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""


def sign_p(b: int, w: int) -> float:
    """Exact one-sided sign test: P(at least b better of b + w)."""
    n = b + w
    return 1.0 if not n else min(1.0, sum(math.comb(n, i) for i in range(b, n + 1)) / 2**n)


def claim_bars(name: str) -> tuple[dict, float, int]:
    """condition -> (% right, CI, stars) from the eval's claim table (pooled), chance 50%."""
    t = load(name)["claim_tables"][0]
    rows = {r["condition"]: r for r in t["rows"] if r["version"] == "pooled"}
    paired = {p["condition"]: p for p in t["paired"]["pooled"] if p["ref"].startswith("transcript_")}
    bars = {c: (r["accuracy"], r["accuracy_ci"], stars(paired.get(c, {}).get("p_one"))) for c, r in rows.items()}
    return bars, 0.5, max(r["n_items"] for r in rows.values())


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a knew case's p > a didn't case's), ties half."""
    d = pos[:, None] - neg[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def auroc_bars(name: str, reps: int = 2000) -> tuple[dict, float, int]:
    """condition -> (AUROC of P(knew), 95% bootstrap CI over cases, stars), chance 0.5 (Kylie, 2026-10-30: in place of % right,
    which rewards a shift in the investigator's bias at the 0.5 threshold). An item's p is the mean over its draws, as
    score.py; stars: one-sided paired bootstrap that the condition's AUROC beats transcript-only's on the same cases."""
    p: dict[str, dict[tuple, list[float]]] = {}
    truth: dict[tuple, int] = {}
    for R in load(name)["runs"]:
        for g in R["groups"]:
            for c, ds in g.get("draws", {}).items():
                for d in ds:
                    k = (g["env"], d["item"])
                    p.setdefault(c, {}).setdefault(k, []).append(d["p"])
                    truth[k] = g["k_cf"]
    keys = sorted(truth)
    y = np.array([truth[k] for k in keys])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(keys), (reps, len(keys)))
    score = lambda x, i: auroc(x[i][y[i] == 1], x[i][y[i] == 0])
    xs = {c: np.array([np.mean(v[k]) for k in keys]) for c, v in p.items()}
    ref = np.array([score(xs["transcript_inline"], i) for i in idx])
    bars = {}
    for c, x in xs.items():
        boot = np.array([score(x, i) for i in idx])
        st = "" if c == "transcript_inline" else stars(float((boot - ref <= 0).mean()))
        bars[c] = (auroc(x[y == 1], x[y == 0]), np.quantile(boot, [0.025, 0.975]).tolist(), st)
    return bars, 0.5, len(keys)


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


def answer_bars(name: str, reps: int = 2000) -> tuple[dict, float, int]:
    """condition -> (% correct, CI, stars) over every item of every run, as the page's headline answer panel."""
    items: dict[tuple, dict[str, list[float]]] = {}
    letters = set()
    for R in load(name)["runs"]:
        if R.get("task") != "answer":
            continue
        for g in R["groups"]:
            for it in g["items"]:
                I = items.setdefault((g["env"], g.get("version") or "", g["base"], g["cf"], it.get("item") or ""), {})
                for c, ds in it["draws"].items():
                    I.setdefault(c, []).extend(float(d["correct"]) for d in ds)
                letters |= {a for a in [it["truth"], *it.get("answers_base", {}), *it.get("answers_cf", {})] if a != "null"}
    per = {c: {k: sum(I[c]) / len(I[c]) for k, I in items.items() if I.get(c)} for c in {c for I in items.values() for c in I}}
    rng = np.random.default_rng(0)
    bars = {}
    for c, xs in per.items():
        x = np.array(list(xs.values()))
        boot = x[rng.integers(0, len(x), (reps, len(x)))].mean(1)
        ref = {k: v - per["transcript_inline"][k] for k, v in xs.items() if k in per["transcript_inline"]}  # this - transcript
        b, w = sum(d > 0 for d in ref.values()), sum(d < 0 for d in ref.values())
        p = None if c == "transcript_inline" else sign_p(b, w)
        bars[c] = (x.mean(), np.quantile(boot, [0.025, 0.975]).tolist(), stars(p))
    chance = 1 / (max(ord(a) for a in letters) - 64)
    return bars, chance, len(items)


def row(axes, evals: dict[str, str], bars_of, chance_label: str, ylabel: str, pct: bool, head: bool, base: float = 0) -> None:
    """One row of the grid, a panel per hint position, in the template's (and the Evals page's answer chart's) style:
    solid bars (placebo hatched, beside its lens), gray 95% CI whisker, the value in ink above it with the stars over
    that, muted group names under the bars, a dotted baseline labelled at the right. `head`: the column titles and case
    counts above the panels. `base`: where the bars and the axis start (Kylie 2026-10-30: 0.5 for AUROC, whose chance
    line is then the axis itself, and 0.4 for % correct)."""
    data = {k: bars_of(name) for k, name in evals.items()}
    for k, name in evals.items():
        data[k][0]["olens_nl_placebo"] = bars_of(name.replace("_tail_inline", "_inline"))[0]["olens_placebo_inline"]
    for ax, (k, (bars, chance, n)) in zip(axes, data.items(), strict=True):
        xs, ticks = positions(bars, GROUPS)
        for c, x in xs.items():
            v, (lo, hi), st = bars[c]
            ax.bar(x, v - base, 0.8, bottom=base, **bar_kw(c))
            ax.errorbar(x, v, [[v - lo], [hi - v]], color=MUTED, lw=0.9, capsize=3, capthick=0.9)
            # offsets in points, so the labels clear the whisker whatever the row's height
            label = f"{v:.1%}" if pct else f"{v:.3f}"
            ax.annotate(label, (x, hi), xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom", fontsize=6.5 * S, color=INK)
            ax.annotate(st, (x, hi), xytext=(0, 6.5), textcoords="offset points", ha="center", va="bottom", fontsize=6.5 * S, color=INK)
        ax.axhline(chance, color=FAINT, lw=0.9, ls=(0, (1, 2)), zorder=1)  # off the axis when above it (letter: 90%)
        ax.set_xticks(ticks, [g for g, cs in GROUPS if any(c in bars for c in cs)])
        ax.set_xlim(-0.6, ticks[-1] + 0.6)
        ax.tick_params(axis="x", labelsize=7 * S)
        if head:
            ax.text(0, 1.2, PANELS[k], transform=ax.transAxes, fontsize=8.5 * S, fontweight="semibold", color=INK)
            ax.text(0, 1.08, f"{n:,} cases · legacy selected cohort", transform=ax.transAxes, fontsize=6.5 * S, color=MUTED)
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
        axes[-1].text(1.0, chance, f" {chance_label}", transform=axes[-1].get_yaxis_transform(), fontsize=7 * S, color=MUTED, va="center")


def unit_mm(fig, ax) -> float:
    """mm per x data unit of `ax` as laid out."""
    fig.canvas.draw()
    return ax.get_window_extent().width / fig.dpi * 25.4 / np.diff(ax.get_xlim())[0]


def grid() -> float:
    """The two questions as rows (did Qwen know it? as AUROC, which letter? as % correct), hint position as columns.
    Returns the mm per bar unit, so other bar charts can match its bar widths and hatching."""
    knew = {"after": "hint_qwen_dsv4_tail_inline", "before": "hint_qwen_dsv4_before_tail_inline"}
    letter = {"after": "hint_qwen_dsv4_letter_tail_inline", "before": "hint_qwen_dsv4_before_letter_tail_inline"}
    widths = [positions(auroc_bars(n)[0] | {"olens_nl_placebo": 0}, GROUPS)[1][-1] + 1 for n in knew.values()]
    fig, axes = figure(FULL_W, 60, nrows=2, ncols=2, sharey="row", gridspec_kw={"width_ratios": widths})
    row(axes[0], knew, auroc_bars, "chance", "did Qwen know it?\nAUROC of P(knew)", pct=False, head=True, base=0.5)
    row(axes[1], letter, hinted_baseline(answer_bars), "always the hinted letter", "which letter?\n% correct",
        pct=True, head=False, base=0.4)  # fmt: skip
    axes[0][-1].text(1.0, 1.08, "hatched = placebo", transform=axes[0][-1].transAxes, fontsize=6.5 * S, color=MUTED, ha="right")
    mm = unit_mm(fig, axes[0][0])
    save(fig, OUT / "grid.pdf")
    return mm


# Part 2, essay grading (cfbench/evals/essay_grading_cf_inline): the investigator's per-case p from the two runs'
# predictions (data/runs/cfbench/, or HF's runs/cfbench/), a case = (arm, intro), its p the mean of its draws, as score.py
ESSAY_RUNS = ["2026-10-30_essay-cf-haiku-ts", "2026-10-30_essay-cf-haiku-olens"]
# Standardized zero-counterfactual OOF probes, nested two-fold, with both positions shown.
ESSAY_INVESTIGATORS = ["transcript_inline", "olens_inline", "olens_placebo_inline"]
ESSAY_GROUPS = [
    ("transcript", ["transcript_inline"]),
    ("O-lens", ["olens_inline", "olens_placebo_inline"]),
    ("transcript\n+ bet at mean", ["transcript_inline+mean"]),
    ("O-lens\n+ bet at mean", ["olens_inline+mean", "olens_placebo_inline+mean"]),
    ("probe:\nuser end", ["probe_user_end"]),
    ("probe:\nfinal", ["probe_response_start"]),
]  # † Kylie: probably cursed (the slide's footnote)


def rank_auroc(p: np.ndarray, y: np.ndarray) -> float:
    """AUROC by ranks, ties half (score.py's)."""
    from scipy.stats import rankdata

    n1 = y.sum()
    return float((rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


@cache
def essay_cases() -> tuple[list[tuple], np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """keys (arm, intro), truth (base mean > arm mean), condition -> per-case score, the observed arm mean."""
    ps: dict[str, dict[tuple, list[float]]] = {}
    truth, arm_mean = {}, {}
    for run in ESSAY_RUNS:
        for r in essay_lens_rows(run):
            if r.get("valid", True) and r.get("p") is not None:
                k = (r["base_arm"], r["item"])
                ps.setdefault(r["condition"], {}).setdefault(k, []).append(r["p"])
                truth[k], arm_mean[k] = r["k_cf"], mean(r["answers_base"])  # answers_base: the observed (arm) prompt's scores
    keys = sorted(truth)
    xs = {c: np.array([np.mean(d[k]) for k in keys]) for c, d in ps.items()}
    for site in ("user_end", "response_start"):
        pr = essay_probe_scores(site)
        assert all(pr[k]["truth"] == truth[k] for k in keys)
        xs[f"probe_{site}"] = np.array([pr[k]["predicted_shift"] for k in keys])
    return keys, np.array([truth[k] for k in keys]), xs, np.array([arm_mean[k] for k in keys])


def _logistic(x: np.ndarray, y: np.ndarray, l2: float = 1e-3) -> np.ndarray:
    w = np.zeros(x.shape[1])
    for _ in range(50):  # Newton
        p = 1 / (1 + np.exp(-x @ w))
        w -= np.linalg.solve(x.T @ (x * (p * (1 - p))[:, None]) + l2 * np.eye(len(w)), x.T @ (p - y) + l2 * w)
    return w


def recalibrated(keys: list[tuple], y: np.ndarray, p: np.ndarray, am: np.ndarray) -> np.ndarray:
    """The investigator plus bet at the mean, as the essay eval page's combined() (branch claude/essay-grading-probe,
    score.py): out-of-fold logistic regression on [1, logit(p), the observed arm mean grade], 5 folds grouped by essay."""
    lp = np.log(np.clip(p, 0.01, 0.99) / (1 - np.clip(p, 0.01, 0.99)))
    x = np.column_stack([np.ones(len(p)), lp, am])
    essays = sorted({k[1] for k in keys})
    fold = {e: i % 5 for i, e in enumerate(np.random.default_rng(0).permutation(essays))}
    f = np.array([fold[k[1]] for k in keys])
    out = np.zeros(len(p))
    for i in range(5):
        out[f == i] = x[f == i] @ _logistic(x[f != i], y[f != i].astype(float))
    return out


@cache
def essay_auroc_bars(reps: int = 2000) -> tuple[dict, float, int, dict]:
    """condition -> (AUROC, 95% CI resampling essays as score.py does, stars: one-sided paired bootstrap vs transcript-only);
    the null (minus the observed arm mean: regression to the mean); n cases; paired differences -> (est, CI, one-sided p)."""
    keys, y, xs, am = essay_cases()
    xs = xs | {f"{c}+mean": recalibrated(keys, y, xs[c], am) for c in ESSAY_INVESTIGATORS}
    items = sorted({k[1] for k in keys})
    of = {i: np.array([n for n, k in enumerate(keys) if k[1] == i]) for i in items}
    rng = np.random.default_rng(0)
    idx = [np.concatenate([of[items[j]] for j in rng.integers(0, len(items), len(items))]) for _ in range(reps)]
    boot = {c: np.array([rank_auroc(x[i], y[i]) for i in idx]) for c, x in xs.items()}
    bars = {}
    for c, x in xs.items():
        ref = "transcript_inline+mean" if c.endswith("+mean") else "transcript_inline"
        st = "" if c == ref else stars(float((boot[c] - boot[ref] <= 0).mean()))
        bars[c] = (rank_auroc(x, y), np.quantile(boot[c], [0.025, 0.975]).tolist(), st)
    pairs = {}
    for a, b in [("olens_inline", "transcript_inline"), ("olens_inline", "olens_placebo_inline"),
                 ("olens_placebo_inline", "transcript_inline"),
                 ("probe_user_end", "transcript_inline"), ("probe_response_start", "transcript_inline"),
                 ("olens_inline+mean", "transcript_inline+mean"), ("olens_inline+mean", "olens_placebo_inline+mean"),
                 ("olens_placebo_inline+mean", "transcript_inline+mean")]:  # fmt: skip
        if a in boot:
            d = boot[a] - boot[b]
            pairs[a, b] = (bars[a][0] - bars[b][0], np.quantile(d, [0.025, 0.975]).tolist(), float((d <= 0).mean()))
    return bars, rank_auroc(-am, y), len(keys), pairs


def essay_flips() -> tuple[int, int]:
    """Cases O-lens gets right (p >= 0.5 vs truth) and its placebo wrong, and the reverse."""
    _, y, xs, _ = essay_cases()
    ok = {c: (xs[c] >= 0.5) == y for c in ("olens_inline", "olens_placebo_inline")}
    return int((ok["olens_inline"] & ~ok["olens_placebo_inline"]).sum()), int((~ok["olens_inline"] & ok["olens_placebo_inline"]).sum())


def essay(mm: float) -> None:
    """AUROC of P(the score goes up without the opener), in grid()'s style, the dotted line at the regression-to-the-mean null.
    `mm`: grid()'s mm per bar unit; the figure is narrowed until its axes match it, so the bars (and their hatching) are
    the same size as Part 1's (Kylie, 2026-10-31)."""
    bars, null, n, _ = essay_auroc_bars()
    fig, ax = figure(90, FULL_H)
    xs, ticks = positions(bars, ESSAY_GROUPS)
    base = 0.5
    for c, x in xs.items():
        v, (lo, hi), st = bars[c]
        ax.bar(x, v - base, 0.8, bottom=base, **bar_kw(c.removesuffix("+mean")))
        ax.errorbar(x, v, [[v - lo], [hi - v]], color=MUTED, lw=0.9, capsize=3, capthick=0.9)
        ax.annotate(f"{v:.3f}", (x, hi), xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom", fontsize=6.5 * S, color=INK)
        ax.annotate(st, (x, hi), xytext=(0, 6.5), textcoords="offset points", ha="center", va="bottom", fontsize=6.5 * S, color=INK)
    ax.axhline(null, color=FAINT, lw=0.9, ls=(0, (1, 2)), zorder=1)
    ax.text(1.0, null, " bet at\n the mean", transform=ax.get_yaxis_transform(), fontsize=7 * S, color=MUTED, va="center")
    ax.set_xticks(ticks, [g for g, cs in ESSAY_GROUPS if any(c in bars for c in cs)])
    ax.set_xlim(-0.6, ticks[-1] + 0.6)
    ax.tick_params(axis="x", labelsize=7 * S)
    ax.text(0, 1.08, f"{n:,} cases · probes: nested 2 × 2", transform=ax.transAxes, fontsize=6.5 * S, color=MUTED)
    ax.text(1.0, 1.08, "hatched = legacy placebo (contaminated)", transform=ax.transAxes, fontsize=6.5 * S, color=MUTED, ha="right")
    ax.grid(axis="x", visible=False)
    top = max(hi for _, (_, hi), _ in bars.values()) + 0.07
    yt = np.arange(base, top, 0.1)
    ax.set_ylim(base, top)
    ax.set_yticks(yt, [f"{t:.1f}" for t in yt])
    ax.set_ylabel("AUROC of P(score goes up)")
    for _ in range(3):  # constrained layout moves the axes as the figure narrows
        w = fig.get_figwidth() - (unit_mm(fig, ax) - mm) * np.diff(ax.get_xlim())[0] / 25.4
        fig.set_figwidth(w)
    save(fig, OUT / "essay.pdf")
    essay_tables()


def essay_diffs() -> None:
    """The paired differences as a forest plot in diff_rows' style (ink CI line and dot, filled when the CI excludes 0, a line
    at 0), compact for the slide's side column: the table above it carries the numbers and swatches. The same essay
    resamples as the table and the stars; the probes are left out so it doesn't set the axis."""
    pairs = essay_auroc_bars()[3]
    short = {"olens_inline": "O-lens", "olens_placebo_inline": "placebo", "transcript_inline": "transcript"}
    fig, ax = figure(57, 30)
    ticks, labels, y = [], [], 0.0
    for head, sfx in (("investigator", ""), ("+ bet at mean", "+mean")):
        ticks.append(y)
        labels.append(head)
        y -= 1
        for a, b in [
            ("olens_inline", "transcript_inline"),
            ("olens_inline", "olens_placebo_inline"),
            ("olens_placebo_inline", "transcript_inline"),
        ]:
            d, (lo, hi), _ = pairs[a + sfx, b + sfx]
            ax.plot([lo, hi], [y, y], color=INK, lw=1.2, solid_capstyle="butt", zorder=3)
            ax.plot(d, y, "o", ms=3.5, zorder=4, mec=INK, mfc=INK if lo > 0 or hi < 0 else BG)
            ticks.append(y)
            labels.append(f"{short[a]} − {short[b]}")
            y -= 1
    ax.set_yticks(ticks, labels)
    for tick, lab in zip(ax.get_yticklabels(), labels, strict=True):
        if lab in ("investigator", "+ bet at mean"):  # group headers, as diff_rows
            tick.set_fontweight("bold")
    ax.axvline(0, color=MUTED, lw=0.8, zorder=2)
    ax.set_ylim(y + 0.4, 0.6)
    ax.set_xlim(-0.012, 0.035)
    ax.set_xticks([0, 0.01, 0.02, 0.03], ["0", "+0.01", "+0.02", "+0.03"])
    ax.set_xlabel("Δ AUROC, paired 95% CI")
    ax.grid(axis="y", visible=False)
    save(fig, OUT / "essay_diffs.pdf")


def essay_tables() -> None:
    """The essay results slide's tables as LaTeX (fig/essay_pairs.tex) in the theme's macros, and the
    O-lens-vs-placebo flip counts for the footnote (fig/essay_flips.tex)."""
    pairs = essay_auroc_bars()[3]
    name = {"transcript_inline": r"\sw{transcript}\ transcript", "olens_inline": r"\sw{olens}\ O-lens",
            "olens_placebo_inline": r"\swp{olens}\ placebo",
            "probe_user_end": r"\sw{probe}\ probe: user end", "probe_response_start": r"\sw{probe}\ probe: final"}  # fmt: skip
    pf = lambda p: "< .001" if p < 1e-3 else f"{p:.3f}".lstrip("0")
    cell = lambda k: f"{pairs[k][0]:+.3f} & {pf(pairs[k][2])}" if k in pairs else "&"
    rows = [f"{name[a]} − {name[b].split(' ', 1)[1]} & {cell((a, b))} & {cell((f'{a}+mean', f'{b}+mean'))}\\\\"
            for a, b in pairs if not a.endswith("+mean")]  # fmt: skip
    tab = lambda spec, lines: f"\\begin{{tabular}}{{@{{}}{spec}@{{}}}}\n" + "\n".join(lines) + "\n\\end{tabular}\n"
    head = [r" & \multicolumn{2}{c}{investigator} & \multicolumn{2}{c}{+ bet at mean}\\", r" & $\Delta$ & p & $\Delta$ & p\\"]
    (OUT / "essay_pairs.tex").write_text(tab("lrrrr", [*head, "\\hline", *rows]))
    (OUT / "essay_flips.tex").write_text("{} and {}".format(*essay_flips()))


def grader() -> None:
    """Two pairs of bars at the "." ending "If you are not sure, give your best guess.", chat template vs the submit_answer tool
    template, on the 863 cutoff_qa items read in both: the share of O-lens reads (six layers) saying "grader"/"graded" (not in the
    prompt), and the share of items with at least one such read, in a lighter shade of the bar's colour (hatching means placebo)."""
    run = Path("runs/cutoff_grader_mentions/2026-10-30_regex")
    if not (ROOT / "data" / run / "novel_hits.jsonl").exists():
        for f in ("item_counts.csv", "novel_hits.jsonl"):
            hf_hub_download(HF_REPO, f"{run}/{f}", repo_type="dataset", local_dir=ROOT / "data")
    groups = [("% of reads", at_token(ROOT / "data" / run, "."), 1.0),
              ("% of items, any read", at_token(ROOT / "data" / run, ".", items=True), 0.5)]  # fmt: skip
    fig, ax = figure(42, 40)
    for g, (_, bars, strength) in enumerate(groups):
        for j, r in enumerate(("A", "B")):
            x, (v, lo, hi) = g * 2.4 + j * 0.85, bars[r]
            ax.bar(x, v, 0.8, color=to_hex(tuple(strength * np.array(to_rgb(RUNS[r][0])) + (1 - strength) * np.array(to_rgb(BG)))))
            ax.errorbar(x, v, [[v - lo], [hi - v]], color=MUTED, lw=0.9, capsize=3, capthick=0.9)
            ax.annotate(
                f"{v:.1%}", (x, hi), xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom", fontsize=6.5 * S, color=INK
            )
    ax.set_xticks([0.425, 2.825], [lab for lab, _, _ in groups])
    ax.set_xlim(-0.6, 3.85)
    ax.tick_params(axis="x", labelsize=7 * S)
    ax.set_ylim(0, 0.85)
    ax.set_yticks(np.arange(0, 0.81, 0.2), [f"{t:.0%}" for t in np.arange(0, 0.81, 0.2)])
    ax.set_ylabel('saying "grader" / "graded"')
    ax.grid(axis="x", visible=False)
    save(fig, OUT / "grader.pdf")


# Part 3, blog authors (cfbench/evals/author_blog_drop_inline): does deleting this sentence make Qwen much less sure of its
# answer about the blogger? Per case the investigator's p (one draw) from the eval's two runs, the task-fit standard L2
# probe's out-of-fold score (scripts/author_probe/period_check.py, runs/author_mcq_blog/2026-10-31_period-probe/scores_l2.jsonl);
# CIs and stars resample bloggers.
AUTHOR_RUNS = ["2026-10-30_author-blog-drop-haiku-b2", "2026-10-31_author-blog-drop-haiku-nla"]
PROBE_RUN = Path("runs/author_mcq_blog/2026-10-31_period-probe")
AUTHOR_CONDS = ["transcript_inline", "olens_inline", "olens_placebo_inline", "nla_inline", "nla_placebo_inline"]
AUTHOR_GROUPS = [("transcript", ["transcript_inline"]), ("O-lens", ["olens_inline", "olens_placebo_inline"]),
                 ("NLA", ["nla_inline", "nla_placebo_inline"]), ("probe", ["probe"]),
                 ("probe +\ntranscript", ["transcript+probe"])]  # fmt: skip
VERSIONS = {"pooled": "all cases"}  # pooled (Kylie, 2026-10-31: not the Qwen-wrong / Qwen-right split)


def author_rows(run: str) -> list[dict]:
    cache = ROOT / "data/plot_cache" / f"{run}.jsonl"
    if cache.exists():
        return [json.loads(line) for line in cache.open()]
    local = ROOT / "data/runs/cfbench" / run / "predictions.jsonl"
    url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/runs/cfbench/{run}/predictions.jsonl"
    fields = ("condition", "item", "pair", "version", "valid", "p", "k_cf")
    with local.open() if local.exists() else urllib.request.urlopen(url) as stream:
        rows = [{k: r[k] for k in fields if k in r} for r in map(json.loads, stream)]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def probe_file(name: str) -> Path:
    f = ROOT / "data" / PROBE_RUN / name
    if not f.exists():
        hf_hub_download(HF_REPO, f"{PROBE_RUN}/{name}", repo_type="dataset", local_dir=ROOT / "data")
    return f


@cache
def author_cases() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """items, truth, version, blogger, condition -> score (the cases with every condition and a probe score)."""
    ps: dict[str, dict[str, float]] = {}
    meta = {}
    for run in AUTHOR_RUNS:
        for r in author_rows(run):
            if r.get("valid", True) and r.get("p") is not None:
                ps.setdefault(r["condition"], {})[r["item"]] = r["p"]
                meta[r["item"]] = (r["k_cf"], r["version"], r["pair"].split(":")[0])
    l2 = {r["item"]: r for r in map(json.loads, probe_file("scores_l2.jsonl").open())}
    probe = {i: r["probe_l2_h"] for i, r in l2.items()}
    items = sorted(i for i in meta if all(i in ps[c] for c in AUTHOR_CONDS) and i in probe)
    xs = {c: np.array([ps[c][i] for i in items]) for c in AUTHOR_CONDS} | {"probe": np.array([probe[i] for i in items])}
    xs["transcript+probe"] = np.array([l2[i]["haiku_plus_probe"] for i in items])  # the nested combiner, out of fold
    y, v, b = (np.array([meta[i][k] for i in items]) for k in range(3))
    return items, y, v, b, xs


@cache
def author_bars(reps: int = 2000) -> tuple[dict, dict, int]:
    """(version, condition) -> (AUROC, 95% CI, stars vs transcript-only, one-sided paired bootstrap over bloggers); paired
    differences (a, b) -> (AUROC est, 95% CI, one-sided p), pooled; n cases."""
    items, y, v, b, xs = author_cases()
    rng = np.random.default_rng(0)
    bars = {}
    for ver in VERSIONS:
        m = np.ones(len(y), bool) if ver == "pooled" else v == ver
        us = sorted(set(b[m]))
        of = {u: np.flatnonzero(m & (b == u)) for u in us}
        idx = [np.concatenate([of[us[j]] for j in rng.integers(0, len(us), len(us))]) for _ in range(reps)]
        boot = {c: np.array([rank_auroc(x[i], y[i]) for i in idx]) for c, x in xs.items()}
        for c, x in xs.items():
            st = "" if c == "transcript_inline" else stars(float((boot[c] - boot["transcript_inline"] <= 0).mean()))
            bars[ver, c] = (rank_auroc(x[m], y[m]), np.quantile(boot[c], [0.025, 0.975]).tolist(), st)
    us = sorted(set(b))
    of = {u: np.flatnonzero(b == u) for u in us}
    idx = [np.concatenate([of[us[j]] for j in rng.integers(0, len(us), len(us))]) for _ in range(reps)]
    pairs = {}
    for a, c in [("olens_inline", "transcript_inline"), ("olens_inline", "olens_placebo_inline"), ("nla_inline", "transcript_inline"),
                 ("nla_inline", "nla_placebo_inline"), ("transcript+probe", "transcript_inline")]:  # fmt: skip
        da = np.array([rank_auroc(xs[a][i], y[i]) - rank_auroc(xs[c][i], y[i]) for i in idx])
        pairs[a, c] = (rank_auroc(xs[a], y) - rank_auroc(xs[c], y), np.quantile(da, [0.025, 0.975]).tolist(), float((da <= 0).mean()))
    return bars, pairs, len(items)


def author() -> None:
    """AUROC of P(the deletion is significant) per condition, pooled over both versions; plus the paired differences table
    (fig/author_pairs.tex)."""
    bars, pairs, n = author_bars()
    one = {c: 0 for _, cs in AUTHOR_GROUPS for c in cs}
    width = positions(one, AUTHOR_GROUPS)[1][-1] + 1.2
    fig, ax0 = figure(70, 58)  # the deck's full figure height (slides/figure.py FULL_H)
    axes = [ax0]
    base = 0.5
    top = max(hi for (_, (_, hi), _) in bars.values()) + 0.06
    for ax, (ver, title) in zip(axes, VERSIONS.items(), strict=True):
        xs, ticks = positions(one, AUTHOR_GROUPS)
        for c, x in xs.items():
            val, (lo, hi), st = bars[ver, c]
            ax.bar(x, val - base, 0.8, bottom=base, **bar_kw("probe_user_end" if c == "transcript+probe" else c))  # a lighter teal
            ax.errorbar(x, val, [[val - lo], [hi - val]], color=MUTED, lw=0.9, capsize=3, capthick=0.9)
            for text, dy in ((f"{val:.3f}", 1.5), (st, 6.5)):
                ax.annotate(
                    text, (x, hi), xytext=(0, dy), textcoords="offset points", ha="center", va="bottom", fontsize=6.5 * S, color=INK
                )
        ax.set_xticks(ticks, [g for g, _ in AUTHOR_GROUPS])
        ax.set_xlim(-0.6, width - 0.6)
        ax.tick_params(axis="x", labelsize=7 * S)
        ax.set_title(title, fontsize=7 * S, color=INK, loc="left")
        ax.grid(axis="x", visible=False)
    yt = np.arange(base, top, 0.1)
    axes[0].set_ylim(base, top)
    axes[0].set_yticks(yt, [f"{t:.1f}" for t in yt])
    axes[0].set_ylabel("AUROC of P(significant)")
    axes[-1].text(1.0, 1.12, "hatched = placebo", transform=axes[-1].transAxes, fontsize=6.5 * S, color=MUTED, ha="right")
    save(fig, OUT / "author.pdf")
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
    essay(grid())
    essay_diffs()
    grader()
    author()
