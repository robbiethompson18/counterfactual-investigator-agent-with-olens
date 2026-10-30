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
# the page's order, with the placebo moved beside its O-lens (the deck's convention); labels as the page's condShort
ORDER = ["transcript_inline", "olens_inline", "olens_placebo_inline", "olens_inline_nl", "probe", "probe_tuned"]
SHORT = {"transcript_inline": "transcript", "olens_inline": "O-lens", "olens_placebo_inline": "placebo",
         "olens_inline_nl": "O-lens\n(nl)", "probe": "probe", "probe_tuned": "tuned\nprobe"}  # fmt: skip
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


def panels(evals: dict[str, str], bars_of, chance_label: str, path: Path) -> None:
    """One panel per eval in the template's (and the Evals page's answer chart's) style: lower-case title with the count
    under it, % gridlines every 20, solid bars (placebo hatched), gray 95% CI whisker, the value in ink above it with the
    stars over that, muted names under the bars, a dotted chance line labelled at the right."""
    data = {k: bars_of(name) for k, name in evals.items()}
    widths = [len(d[0]) for d in data.values()]
    fig, axes = figure(FULL_W, 50, ncols=len(evals), sharey=True, gridspec_kw={"width_ratios": widths})
    for ax, (k, (bars, chance, n)) in zip(axes, data.items(), strict=True):
        conds = [c for c in ORDER if c in bars]
        for x, c in enumerate(conds):
            v, (lo, hi), st = bars[c]
            ax.bar(x, v, 0.66, **bar_kw(c))
            ax.errorbar(x, v, [[v - lo], [hi - v]], color=MUTED, lw=0.9, capsize=3, capthick=0.9)
            ax.text(x, hi + 0.015, f"{v:.0%}", ha="center", va="bottom", fontsize=6.5 * S, color=INK)
            ax.text(x, hi + 0.05, st, ha="center", va="bottom", fontsize=6.5 * S, color=INK)
        ax.axhline(chance, color=FAINT, lw=0.9, ls=(0, (1, 2)), zorder=1)
        ax.set_xticks(range(len(conds)), [SHORT[c] for c in conds])
        ax.set_xlim(-0.6, len(conds) - 0.4)
        ax.tick_params(axis="x", length=0, labelcolor=MUTED, labelsize=7 * S)
        ax.text(0, 1.13, PANELS[k], transform=ax.transAxes, fontsize=8.5 * S, fontweight="semibold", color=INK)
        ax.text(0, 1.04, f"{n:,} cases · {evals[k]}", transform=ax.transAxes, fontsize=6.5 * S, color=MUTED)
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 1)
    axes[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1], ["0%", "20%", "40%", "60%", "80%", "100%"])
    axes[0].tick_params(axis="y", labelcolor=MUTED)
    axes[-1].text(1.0, data[list(evals)[-1]][1], f" {chance_label}", transform=axes[-1].get_yaxis_transform(), fontsize=7 * S,
                  color=MUTED, va="center")  # fmt: skip
    save(fig, path)


if __name__ == "__main__":
    panels({"after": "hint_qwen_dsv4_tail_inline", "before": "hint_qwen_dsv4_before_tail_inline"}, claim_bars, "chance",
           OUT / "knew.pdf")  # fmt: skip
    panels({"after": "hint_qwen_dsv4_letter_tail_inline", "before": "hint_qwen_dsv4_before_letter_tail_inline"}, answer_bars,
           "chance", OUT / "letter.pdf")  # fmt: skip
