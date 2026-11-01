"""menus_inline: which read positions matter. Haiku 5.5 names Qwen3.6-27B's answer from the bare prompt plus one lens's reads
at one menu of positions (cfbench/evals/menus_inline): `prompt` (every punctuation token and option newline up to the
question's final "."), `tail` (the 9 tokens after it, `<|im_end|>` ... the `\\n\\n` after `</think>`), `all` (both) or `last`
(that final `\\n\\n` only), for O-lens, NLA and J++, each beside a same-category placebo (docs/placebo-rules.md), on four
sets: cutoff_qa, favorites_qa, mcq_unselected (MMLU-Pro), SuperGPQA. One draw per case.

docs/img/menus-inline.png: % of cases where Haiku names Qwen's answer, a panel per lens, a group per menu (lens and its hatched
placebo), prompt only as the blue bar and dashed line; each bar is the mean of the four sets' accuracies (each set weighted
equally), 95% CIs from resampling cases within every set.

docs/img/menus-inline-position.png (the write-up's "By token position"): one bar per lens, its `prompt` menu minus its `tail`
menu on the same cases, pooled the same way, with a paired 95% CI and two-sided bootstrap p.

docs/img/menus-inline-by-menu.png (the same section): nine bars, each lens at the `prompt`, `tail` and `all` menus, pooled the same
way, prompt only as the dashed line; the `prompt` and `all` menus have 3 draws per case, `tail` has 1 (noted in the figure).

    uv run python -m graphs.menus_inline [--out docs/img/menus-inline.png]
"""

import argparse
import json
from collections import defaultdict
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cfbench.catalog import CF as R
from cfbench.catalog import MENUS as SETS  # set -> (title, runs)
from cfbench.scoring import prediction_fields
from graphs.style import BLUE, INK, MUTED, OLENS, bar_kw, ci_bar, color, departure

LENSES = {"olens": OLENS, "nla": "NLA", "jlens": "J++"}
MENUS = {"prompt": "prompt punctuation", "tail": "9 tail tokens", "all": "all", "last": "last token"}
BOOT = 2000


def runs(name: str) -> list[Path]:
    """Set `name`'s finished runs."""
    return [
        R / run
        for run in SETS[name][1]
        if (R / run / "meta.json").exists() and json.loads((R / run / "meta.json").read_text()).get("status") == "done"
    ]


@cache
def load() -> dict[str, dict[str, dict[str, float]]]:
    """set -> condition -> case -> share of its draws where Haiku named Qwen's answer, from every run of SETS that has finished."""
    draws: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))  # set -> condition -> case -> draw -> right
    for name in SETS:
        for run in runs(name):
            for r in prediction_fields(run):
                draws[name][r["condition"]][r["item"]][r.get("draw", 0)] = float(bool(r.get("correct")))
    return {s: {c: {i: sum(v.values()) / len(v) for i, v in items.items()} for c, items in conds.items()} for s, conds in draws.items()}


def boot_means(per_set: list[list[float]], seed: int = 0) -> tuple[float, np.ndarray]:
    """Mean over sets of each set's mean, and BOOT resamples of it (cases resampled within every set), vectorised: a multinomial
    count matrix per set times its values."""
    rng = np.random.default_rng(seed)
    boots = np.zeros(BOOT)
    for d in per_set:
        x = np.asarray(d, dtype=float)
        boots += rng.multinomial(len(x), np.full(len(x), 1 / len(x)), size=BOOT) @ x / len(x)
    return float(np.mean([np.mean(d) for d in per_set])), boots / len(per_set)


def pooled_ci(per_set: list[list[float]], seed: int = 0) -> tuple[float, float, float]:
    """Mean over sets of each set's mean, CI from resampling cases within every set."""
    mean, boots = boot_means(per_set, seed)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean, float(lo), float(hi)


def pooled(acc: dict, c: str) -> tuple[float, float, float] | None:
    """Condition c's accuracy (%), the mean over the four sets with each set weighted equally, CI from resampling cases within
    every set; None until every set has c."""
    if not all(c in acc[s] for s in SETS):
        return None
    return pooled_ci([[100 * v for v in acc[s][c].values()] for s in SETS])


def paired(acc: dict, a: str, b: str, seed: int = 0) -> tuple[float, float, float, float]:
    """Condition a minus b in pp on the cases both have: mean over sets (each weighted equally), 95% CI and two-sided p from
    resampling cases within every set."""
    per_set = [[100 * (acc[s][a][k] - acc[s][b][k]) for k in sorted(acc[s][a].keys() & acc[s][b].keys())] for s in SETS]
    mean, boots = boot_means(per_set, seed)
    p = min(1.0, 2 * min(float((boots <= 0).sum()), float((boots >= 0).sum())) / BOOT)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return mean, float(lo), float(hi), p


def position_diff(acc: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.4, 3.4), layout="constrained")
    ticks = []
    for x, (lens, lens_label) in enumerate(LENSES.items()):
        v, lo, hi, p = paired(acc, f"{lens}_prompt_inline", f"{lens}_tail_inline")
        ax.bar(x, v, 0.7, zorder=3, **bar_kw(lens))
        ax.errorbar(x, v, [[v - lo], [hi - v]], color=MUTED, lw=0.9, capsize=3, capthick=0.9, zorder=4)
        ax.annotate(f"{v:+.1f}", (x, hi if v > 0 else lo), xytext=(0, 2 if v > 0 else -2), textcoords="offset points",
                    ha="center", va="bottom" if v > 0 else "top", fontsize=8, color=INK)  # fmt: skip
        ticks.append(f"{lens_label}\np {'< 0.001' if p < 0.001 else f'= {p:.3f}'}")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(range(len(LENSES)), ticks)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(-3, 4)
    ax.set_ylabel("question reads − tail reads (pp)")
    ax.set_title("Reads at the question vs at the 9 tail tokens", loc="left")
    fig.supxlabel("Paired, mean of the four sets (each weighted equally); 95% CIs resample cases within each set.", fontsize=7.5,
                  color=MUTED)  # fmt: skip
    fig.savefig(out, bbox_inches="tight")
    print(out)


def by_menu(acc: dict, out: Path) -> None:
    """Nine bars: each lens at the question reads, the 9 tail tokens and both, pooled over the four sets."""
    menus = {"prompt": "question", "tail": "tail", "all": "both"}
    fig, ax = plt.subplots(figsize=(7.2, 3.6), layout="constrained")
    base = pooled(acc, "transcript_inline")
    assert base
    ax.axhline(base[0], color=BLUE, lw=0.8, ls=(0, (3, 3)), zorder=2)
    ticks, x = [], 0.0
    for lens, lens_label in LENSES.items():
        for menu, menu_label in menus.items():
            v = pooled(acc, f"{lens}_{menu}_inline")
            assert v
            ci_bar(ax, x, lens, *v)
            ticks.append((x, menu_label))
            x += 0.85
        ax.text(x - 0.85 * 2, -0.19, lens_label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9.5, color=INK)
        x += 0.75
    ax.set_xticks([t for t, _ in ticks], [m for _, m in ticks], fontsize=8.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(70, 90)
    ax.set_yticks(range(70, 91, 5))
    ax.set_ylabel("% of cases where Haiku names Qwen's answer")
    ax.set_title("Where to read: the question, the 9 tail tokens, or both", loc="left")
    fig.supxlabel("Mean of the four sets, each weighted equally; 95% CIs resample cases within each set. Dashed = prompt only.",
                  fontsize=7.5, color=MUTED)  # fmt: skip
    departure(fig, "the question and both menus have 3 draws per case; the tail menu has 1.")
    fig.savefig(out, bbox_inches="tight")
    print(out)


THREE = {"prompt": "question", "tail": "tail", "all": "both"}  # the write-up's three places to read, in its words
DRAWS = {"prompt": 3, "tail": 1, "all": 3}  # draws per case in the canonical runs (the `all` and `prompt` top-ups)


def menu_bars(ax, acc: dict, s: float = 1.0) -> None:
    """Nine bars: each lens at the question tokens, the 9 tail tokens and both, pooled over the four sets, prompt only as the
    dashed line (the write-up's "By token position" chart). `s` scales the text (the deck passes graphs.style.SLIDE_S)."""
    base = pooled(acc, "transcript_inline")
    assert base
    ax.axhline(base[0], color=BLUE, lw=0.8, ls=(0, (3, 3)), zorder=2)  # prompt only: the caption says so
    ticks, x = [], 0.0
    for lens, lens_label in LENSES.items():
        for menu, menu_label in THREE.items():
            v = pooled(acc, f"{lens}_{menu}_inline")
            assert v, (lens, menu)
            ci_bar(ax, x, lens, *v, fontsize=7.5 * s)
            ticks.append((x, menu_label))
            x += 0.85
        ax.text(x - 0.85 * 2, 1.01, lens_label, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8.5 * s,
                color=color(lens), weight="bold")  # fmt: skip
        x += 0.6
    ax.set_xticks([t for t, _ in ticks], [m for _, m in ticks], fontsize=7.5 * s)
    ax.set_xlim(-0.6, x - 0.6 - 0.85 + 0.6)
    ax.set_ylim(60, 92)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("% of cases where Haiku names Qwen's answer")


def bars(acc: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, len(LENSES), figsize=(4.6 * len(LENSES), 4.0), sharey=True, layout="constrained")
    base = pooled(acc, "transcript_inline")
    assert base
    for ax, (lens, lens_label) in zip(axes, LENSES.items(), strict=True):
        ci_bar(ax, 0, "prompt", *base)
        ax.axhline(base[0], color=BLUE, lw=0.8, ls=(0, (3, 3)), zorder=2)
        ticks, x = [0.0], 1.45
        for menu in MENUS:
            for i, style in enumerate((lens, f"{lens}_placebo")):  # graphs.style keys
                if v := pooled(acc, f"{lens}_{menu}{'_placebo' * i}_inline"):
                    ci_bar(ax, x + 0.85 * i, style, *v)
            ticks.append(x + 0.425)
            x += 0.85 + 1.45
        ax.set_xticks(ticks, ["prompt\nonly", *(label.replace(" ", "\n", 1) for label in MENUS.values())], fontsize=8.5)
        ax.grid(axis="x", visible=False)
        ax.set_title(lens_label)
        if not pooled(acc, f"{lens}_last_inline"):
            ax.text(0.6, 0.5, f"{lens_label}: not every set done yet", transform=ax.transAxes, ha="center", color=MUTED, fontsize=9)
    axes[0].set_ylim(40, 90)
    axes[0].set_ylabel("% of cases where Haiku names Qwen's answer")
    sets = ", ".join(f"{title} ({len(acc[s]['transcript_inline'])})" for s, (title, _) in SETS.items())
    fig.suptitle("Which read positions help: each lens at four position menus, pooled over four sets", x=0.01, ha="left")
    note = f"Mean of the four sets, each weighted equally: {sets}. Hatched = placebo (another case's reads, same category); dashed ="
    fig.supxlabel(f"{note} prompt only. 95% CIs resample cases within each set.", fontsize=8, color=MUTED)
    fig.savefig(out, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("docs/img/menus-inline.png"))
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    acc = load()
    bars(acc, a.out)
    position_diff(acc, a.out.with_name(a.out.stem + "-position.png"))
    fig, ax = plt.subplots(figsize=(6.4, 3.6), layout="constrained")
    menu_bars(ax, acc)
    ax.set_title("Each lens read at the question, at the 9 tail tokens, and at both", loc="left", y=1.08)
    fig.supxlabel("Mean of the four sets, each weighted equally; 95% CIs resample cases within each set. 3 draws per case at "
                  "question and both, 1 at tail.", fontsize=7.5, color=MUTED)  # fmt: skip
    fig.savefig(a.out.with_name(a.out.stem + "-menus.png"), bbox_inches="tight")
    print(a.out.with_name(a.out.stem + "-menus.png"))
    by_menu(acc, a.out.with_name(a.out.stem + "-by-menu.png"))
