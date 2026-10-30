"""The hint evals' bars, one panel per eval as on its Evals page (#eval=NAME), from the real results: the viewer's
evals/NAME.json (data/evals/, or HF's evals/ when it isn't local).

- Did Qwen know it? (hint_qwen_dsv4{,_before}_tail_inline, claim evals): % right with guess = p >= 0.5 and the 95% CI
  from claim_tables (cfbench/evals/<name>/score.py), stars from its paired one-sided sign test against transcript-only.
- Which letter? (hint_qwen_dsv4{,_before}_letter_tail_inline, answer evals): the page's headline answer panel recomputed
  as page.html's answerChartBlock does: every item's draws merged across runs, a bar is the mean over items of the item's
  fraction right, a 95% item bootstrap (2000 resamples; the page's own RNG differs, so a CI end can move by a point), the
  exact one-sided sign test against transcript-only over the items that differ, chance = 1 / the most options.

    uv run python -m slides.hint_evals.figures   # -> slides/hint_evals/fig/*.pdf
"""

import json
import math
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from graphs.style import INK, MUTED
from slides.figure import FULL_W, bar_kw, figure, save

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("fig")
HF = "https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens/resolve/main/evals/"
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


def panels(evals: dict[str, str], bars_of, chance_label: str, ylabel: str, path: Path, pct: bool = True) -> None:
    """One panel per eval in the template's (and the Evals page's answer chart's) style: lower-case title with the count
    under it, % gridlines every 20, solid bars (placebo hatched), gray 95% CI whisker, the value in ink above it with the
    stars over that, muted names under the bars, a dotted chance line labelled at the right."""
    data = {k: bars_of(name) for k, name in evals.items()}
    for k, name in evals.items():
        data[k][0]["olens_nl_placebo"] = bars_of(name.replace("_tail_inline", "_inline"))[0]["olens_placebo_inline"]
    layout = {k: positions(d[0]) for k, d in data.items()}
    widths = [lay[1][-1] + 1 for lay in layout.values()]
    fig, axes = figure(FULL_W, 50, ncols=len(evals), sharey=True, gridspec_kw={"width_ratios": widths})
    for ax, (k, (bars, chance, n)) in zip(axes, data.items(), strict=True):
        xs, ticks = layout[k]
        for c, x in xs.items():
            v, (lo, hi), st = bars[c]
            ax.bar(x, v, 0.8, **bar_kw(c))
            ax.errorbar(x, v, [[v - lo], [hi - v]], color=MUTED, lw=0.9, capsize=3, capthick=0.9)
            ax.text(x, hi + 0.015, f"{v:.1%}" if pct else f"{v:.3f}", ha="center", va="bottom", fontsize=6.5 * S, color=INK)
            ax.text(x, hi + 0.05, st, ha="center", va="bottom", fontsize=6.5 * S, color=INK)
        ax.axhline(chance, color=FAINT, lw=0.9, ls=(0, (1, 2)), zorder=1)  # off the axis when above it (letter: 90%)
        ax.set_xticks(ticks, [g for g, cs in GROUPS if any(c in bars for c in cs)])
        ax.set_xlim(-0.6, ticks[-1] + 0.6)
        ax.tick_params(axis="x", length=0, labelcolor=MUTED, labelsize=7 * S)
        ax.text(0, 1.13, PANELS[k], transform=ax.transAxes, fontsize=8.5 * S, fontweight="semibold", color=INK)
        ax.text(0, 1.04, f"{n:,} cases", transform=ax.transAxes, fontsize=6.5 * S, color=MUTED)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", visible=False)
    # the axis stops just above the tallest whisker's labels, not at the next round number (Kylie, 2026-10-30); gridlines
    # every 10% below it
    top = max(hi for d in data.values() for _, (_, hi), _ in d[0].values()) + 0.07
    ticks = np.arange(0, top, 0.1)
    axes[0].set_ylim(0, top)
    axes[0].set_yticks(ticks, [f"{t:.0%}" if pct else f"{t:.1f}" for t in ticks])
    axes[0].tick_params(axis="y", labelcolor=MUTED)
    axes[0].set_ylabel(ylabel)
    if data[list(evals)[-1]][1] <= top:
        chance = data[list(evals)[-1]][1]
        axes[-1].text(1.0, chance, f" {chance_label}", transform=axes[-1].get_yaxis_transform(), fontsize=7 * S, color=MUTED, va="center")
    axes[-1].text(1.0, 1.04, "hatched = placebo", transform=axes[-1].transAxes, fontsize=6.5 * S, color=MUTED, ha="right")
    save(fig, path)


def positions(bars: dict) -> tuple[dict[str, float], list[float]]:
    """x of every bar (a lens and its placebo touch, groups 0.6 apart) and the tick under each group's centre."""
    xs, ticks, x = {}, [], 0.0
    for _, cs in GROUPS:
        cs = [c for c in cs if c in bars]
        if not cs:
            continue
        for i, c in enumerate(cs):
            xs[c] = x + 0.85 * i
        ticks.append(x + 0.85 * (len(cs) - 1) / 2)
        x += 0.85 * (len(cs) - 1) + 1.45
    return xs, ticks


if __name__ == "__main__":
    knew = {"after": "hint_qwen_dsv4_tail_inline", "before": "hint_qwen_dsv4_before_tail_inline"}
    letter = {"after": "hint_qwen_dsv4_letter_tail_inline", "before": "hint_qwen_dsv4_before_letter_tail_inline"}
    panels(knew, auroc_bars, "chance", "AUROC of P(knew)", OUT / "knew.pdf", pct=False)
    panels(letter, hinted_baseline(errors(answer_bars)), "always the hinted letter", "error rate (lower is better)", OUT / "letter.pdf")
