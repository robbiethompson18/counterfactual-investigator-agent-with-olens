"""hint_unselected: did Qwen know it, on the unselected hinted-MCQ set (cfbench/evals/hint_unselected_inline, docs/hint-unselected.md).

Two panels: AUROC of P(knew) (chance 0.5) and % correct on the letter task (dotted line: always the hinted letter, 50%), each
with transcript only and the O-lens at the `prompt` and `all` menus, each beside its
hatched placebo; 95% bootstrap CIs over cases, draws pooled (docs/cfbench-scoring.md);
stars: one-sided paired bootstrap of the O-lens against transcript only (over the O-lens bar) and against its placebo (over
the placebo bar), from the evals' score modules.

    uv run python -m graphs.hint_unselected   # -> docs/img/hint-unselected.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

from cfbench.evals.hint_unselected_inline import score as knew_score
from cfbench.evals.hint_unselected_letter_inline import score as letter_score
from cfbench.scoring import stars
from graphs.style import INK, MUTED, bar_kw, positions

R = Path("data/runs/cfbench")
RUNS = {"knew": ["2026-10-31_hint-unselected", "2026-10-31_hint-unselected-all"],
        "letter": ["2026-10-31_hint-unselected-letter", "2026-10-31_hint-unselected-letter-all"]}  # fmt: skip
GROUPS = [("transcript", ["transcript_inline"]), ("O-lens\nprompt reads", ["olens_prompt_inline", "olens_prompt_placebo_inline"]),
          ("O-lens\n+ tail reads", ["olens_all_inline", "olens_all_placebo_inline"])]  # fmt: skip
OUT = Path("docs/img/hint-unselected.png")


def rows(task: str) -> list[dict]:
    return [json.loads(line) for run in RUNS[task] for line in (R / run / "predictions.jsonl").open()]


def panel(ax, task: str, mod, key: str, base: float, pct: bool, ylabel: str, chance: float, chance_label: str) -> None:
    rs = rows(task)
    bars = {s["condition"]: (s[key], s["ci"]) for s in mod.stats(rs)}
    p = {(x["a"], x["b"]): x["p_one_sided"] for x in mod.paired(rs)}
    xs, ticks = positions(bars, GROUPS)
    for c, x in xs.items():
        v, (lo, hi) = bars[c]
        ax.bar(x, v - base, 0.8, bottom=base, **bar_kw("olens_placebo" if "placebo" in c else "olens" if "olens" in c else c))
        ax.errorbar(x, v, [[v - lo], [hi - v]], color=MUTED, lw=0.9, capsize=3, capthick=0.9)
        lens = c.replace("_placebo", "")
        st = stars(p.get((lens, "transcript_inline")) if c == lens else p.get((lens, c)))
        ax.annotate((f"{v:.1%}" if pct else f"{v:.3f}") + (f"\n{st}" if st else ""), (x, hi), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, color=INK)  # fmt: skip
    ax.axhline(chance, color=MUTED, lw=0.9, ls=(0, (1, 2)), zorder=1)
    ax.text(1.0, chance, f" {chance_label}", transform=ax.get_yaxis_transform(), fontsize=7, color=MUTED, va="center")
    ax.set_xticks(ticks, [g for g, _ in GROUPS])
    ax.set_xlim(-0.6, ticks[-1] + 0.6)
    top = max(hi for _, (_, hi) in bars.values()) + (0.12 if pct else 0.08)
    ax.set_ylim(base, top)
    ax.set_ylabel(ylabel)
    if pct:
        ax.set_yticks(
            [t / 10 for t in range(int(base * 10), int(top * 10) + 1)],
            [f"{t:.0%}" for t in [t / 10 for t in range(int(base * 10), int(top * 10) + 1)]],
        )
    n = len({r["item"] for r in rs})
    ax.set_title(f"{n:,} cases, draws pooled", fontsize=8, color=MUTED, loc="left")
    ax.grid(axis="x", visible=False)


def main(out: Path = OUT) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), layout="constrained")
    panel(axes[0], "knew", knew_score, "auroc", 0.5, False, "did Qwen know it?\nAUROC of P(knew)", 0.5, "chance")
    panel(
        axes[1], "letter", letter_score, "accuracy", 0.4, True, "which letter without the hint?\n% correct", 0.5, "always the hinted letter"
    )
    fig.suptitle("Hinted MMLU-Pro, unselected (consistency filter only), hint first, Haiku 5.5, canonical reads", fontsize=9, color=INK)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
