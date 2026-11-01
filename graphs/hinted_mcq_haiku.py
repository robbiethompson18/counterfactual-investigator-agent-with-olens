"""Hinted MCQ, hint-first cases, by investigator (docs/final-writeup.md, Hinted MCQ): the grid chart's Gemini 3.8 Flash at effort
low beside the same cases rerun with the default investigator, Haiku 5.5 at medium. Left, AUROC of P(knew); right, % correct on
Qwen's no-hint letter; transcript only, the O-lens and its placebo for each. Flash's bars are cfbench.scoring's auroc_bars /
answer_bars on the Evals-page evals, Haiku's hint_haiku_bars on the rerun (cfbench/catalog.py HINT_FLASH_EVALS, HINT_HAIKU_RUNS).

    uv run python -m graphs.hinted_mcq_haiku   # -> docs/img/hinted-mcq-haiku.png
"""

from pathlib import Path

import matplotlib.pyplot as plt

from cfbench.catalog import HINT_FLASH_EVALS
from cfbench.scoring import answer_bars, auroc_bars, hint_haiku_bars
from graphs.style import FAINT, INK, MUTED, OLENS, bar_kw, departure, label

CONDS = ["transcript_inline", "olens_inline", "olens_placebo_inline"]
INVESTIGATORS = ["Gemini 3.8 Flash\neffort low", "Haiku 5.5\neffort medium"]
DEPARTURE = (f"hint-first cases only; {OLENS} at layers 32/40/48/52; positions are the option newlines and the end of the prompt "
             "only; the left group is the grid chart's investigator, Gemini 3.8 Flash at effort low, the right our default, "
             "Haiku 5.5 at effort medium, 3 draws.")  # fmt: skip


def main() -> None:
    haiku = hint_haiku_bars()
    panels = {
        "knew": ("did Qwen know it?\nAUROC of P(knew)", [auroc_bars(HINT_FLASH_EVALS["knew"]), haiku["knew"]], "{:.3f}", 0.5, "chance"),
        "letter": ("which letter without the hint?\n% correct", [answer_bars(HINT_FLASH_EVALS["letter"]), haiku["letter"]], "{:.1%}", 0.5,
                   "always the hinted letter"),
    }  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9), layout="constrained")
    for ax, (ylabel, data, fmt, base, base_label) in zip(axes, panels.values(), strict=True):
        ticks = []
        for g, (bars, _, n) in enumerate(data):
            assert n == data[0][2], "both investigators must see the same cases"
            x0 = g * 4.0
            for i, c in enumerate(CONDS):
                v, (lo, hi), st = bars[c]
                x = x0 + (0, 1.3, 2.15)[i]  # the lens and its placebo touch
                ax.bar(x, v - base, 0.8, bottom=base, zorder=3, **bar_kw(c))
                ax.errorbar(x, v, [[v - lo], [hi - v]], color=MUTED, lw=0.9, capsize=3, capthick=0.9, zorder=4)
                for text, dy in ((fmt.format(v), 1.5), (st, 7.5)):
                    ax.annotate(
                        text, (x, hi), xytext=(0, dy), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, color=INK
                    )
            ticks.append(x0 + 2.15 / 2)
        ax.axhline(base, color=FAINT, lw=0.9, ls=(0, (1, 2)), zorder=1)
        ax.text(1.0, base, f" {base_label}", transform=ax.get_yaxis_transform(), fontsize=7.5, color=MUTED, va="center")
        ax.set_xticks(ticks, INVESTIGATORS, fontsize=9)
        ax.set_xlim(-0.6, ticks[-1] + 2.15 / 2 + 0.6)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0.5, 0.8)
    axes[1].set_ylim(0.4, 0.7)
    axes[1].set_yticks([0.4, 0.5, 0.6, 0.7], ["40%", "50%", "60%", "70%"])
    n = panels["knew"][1][0][2]
    fig.suptitle(f"Hinted MCQ, hint first: the {OLENS} gain is the investigator's", x=0.01, ha="left")
    fig.supxlabel(f"{n:,} cases · 95% bootstrap CIs over cases · stars: one-sided paired test against transcript only · "
                  f"{label('olens_inline')} beside its hatched placebo", fontsize=8, color=MUTED)  # fmt: skip
    departure(fig, DEPARTURE)
    out = Path("docs/img/hinted-mcq-haiku.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    for k, (_, data, *_) in panels.items():
        print(k, [{c: f"{bars[c][0]:.3f}{bars[c][2]}" for c in CONDS} for bars, _, _ in data])
    print("wrote", out)


if __name__ == "__main__":
    main()
