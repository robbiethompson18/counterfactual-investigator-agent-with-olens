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

    uv run python -m graphs.menus_inline [--out docs/img/menus-inline.png]
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from graphs.style import BLUE, INK, MUTED, bar_kw, ci_bar

R = Path("data/runs/cfbench")
# set -> (title, runs); a set's NLA run is left out until it exists
SETS = {
    "cutoff": ("cutoff_qa", ["2026-10-30_menus-cutoff-olens-jlens", "2026-10-31_menus-cutoff-nla-batch"]),
    "favorites": ("favorites_qa", ["2026-10-30_menus-favorites-olens-jlens", "2026-10-31_menus-favorites-nla-batch"]),
    "mcq_unselected": ("MMLU-Pro (mcq_unselected)", ["2026-10-31_menus-mcq-unselected-olens-jlens-batch",
                                                     "2026-10-31_menus-mcq-unselected-nla-or2"]),
    "supergpqa": ("SuperGPQA", ["2026-10-31_menus-supergpqa-olens-jlens-batch2", "2026-10-31_menus-supergpqa-nla-batch3"]),
}  # fmt: skip
LENSES = {"olens": "O-lens", "nla": "NLA", "jlens": "J++"}
MENUS = {"prompt": "prompt punctuation", "tail": "9 tail tokens", "all": "all", "last": "last token"}
BOOT = 2000


def load() -> dict[str, dict[str, dict[str, float]]]:
    """set -> condition -> case -> 1.0 if Haiku named Qwen's answer, from every run of SETS that has finished."""
    acc: dict = defaultdict(lambda: defaultdict(dict))
    for name, (_, runs) in SETS.items():
        for run in runs:
            meta, rows = R / run / "meta.json", R / run / "predictions.jsonl"
            if not meta.exists() or json.loads(meta.read_text()).get("status") != "done":
                continue
            for r in map(json.loads, rows.open()):
                acc[name][r["condition"]][r["item"]] = float(bool(r.get("correct")))
    return acc


def pooled_ci(per_set: list[list[float]], seed: int = 0) -> tuple[float, float, float]:
    """Mean over sets of each set's mean, CI from resampling cases within every set."""
    rng = random.Random(seed)
    boots = sorted(sum(sum(rng.choices(d, k=len(d))) / len(d) for d in per_set) / len(per_set) for _ in range(BOOT))
    return sum(sum(d) / len(d) for d in per_set) / len(per_set), boots[int(0.025 * BOOT)], boots[int(0.975 * BOOT)]


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
    rng = random.Random(seed)
    boots = sorted(sum(sum(rng.choices(d, k=len(d))) / len(d) for d in per_set) / len(per_set) for _ in range(BOOT))
    p = min(1.0, 2 * min(sum(x <= 0 for x in boots), sum(x >= 0 for x in boots)) / BOOT)
    return sum(sum(d) / len(d) for d in per_set) / len(per_set), boots[int(0.025 * BOOT)], boots[int(0.975 * BOOT)], p


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


def bars(acc: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, len(LENSES), figsize=(4.6 * len(LENSES), 4.0), sharey=True, layout="constrained")
    base = pooled(acc, "transcript_inline")
    assert base
    for ax, (lens, lens_label) in zip(axes, LENSES.items(), strict=True):
        ci_bar(ax, 0, "transcript", *base)
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
