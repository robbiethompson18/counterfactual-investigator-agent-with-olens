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
import numpy as np

from cfbench.evals.hint_unselected_inline import score as knew_score
from cfbench.evals.hint_unselected_inline.score import auroc
from cfbench.evals.hint_unselected_letter_inline import score as letter_score
from cfbench.scoring import stars
from graphs.style import INK, MUTED, bar_kw, positions

R = Path("data/runs/cfbench")
RUNS = {"knew": ["2026-10-31_hint-unselected-d", "2026-10-31_hint-unselected-all-d"],
        "letter": ["2026-10-31_hint-unselected-letter-d", "2026-10-31_hint-unselected-letter-all-d"]}  # fmt: skip
GROUPS = [("transcript", ["transcript_inline"]), ("O-lens\nprompt reads", ["olens_prompt_inline", "olens_prompt_placebo_inline"]),
          ("O-lens\n+ tail reads", ["olens_all_inline", "olens_all_placebo_inline"])]  # fmt: skip
OUT = Path("docs/img/hint-unselected.png")


# the legacy DSv4-disagreement cohort rerun with this eval's investigator (Haiku 5.5 at medium, 3 draws) on its own reads
# (cfbench/evals/hint_qwen_dsv4_before{,_letter}_tail_inline, robbie-box, 2026-10-31 evening): the talk grid's middle column
LEGACY_HAIKU = {"knew": ["2026-11-01_hint-dsv4-before-haiku"], "letter": ["2026-11-01_hint-dsv4-before-letter-haiku"]}


def rows(task: str, runs: list[str] | None = None) -> list[dict]:
    """Every valid prediction row of the task's runs (a draw with no valid submission has p / correct null)."""
    key = "p" if task == "knew" else "correct"
    out = [json.loads(line) for run in runs or RUNS[task] for line in (R / run / "predictions.jsonl").open()]
    return [r for r in out if r.get(key) is not None]


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
    n_of = {c: len({r["item"] for r in rs if r["condition"] == c}) for c in bars}  # a pair short of cases says so
    n_all = max(n_of.values())
    ax.set_xticks(
        ticks, [g + (f"\n({min(n_of[c] for c in cs):,} cases)" if min(n_of[c] for c in cs) < 0.9 * n_all else "") for g, cs in GROUPS]
    )
    ax.set_xlim(-0.6, ticks[-1] + 0.6)
    top = max(hi for _, (_, hi) in bars.values()) + (0.12 if pct else 0.08)
    ax.set_ylim(base, top)
    ax.set_ylabel(ylabel)
    if pct:
        ax.set_yticks(
            [t / 10 for t in range(int(base * 10), int(top * 10) + 1)],
            [f"{t:.0%}" for t in [t / 10 for t in range(int(base * 10), int(top * 10) + 1)]],
        )
    ax.set_title(f"{n_all:,} cases, draws pooled", fontsize=8, color=MUTED, loc="left")
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


# the talk's and write-up's two-cohort grid (graphs.talk grid): the canonical `prompt` menu only, under the legacy evals' condition
# names so the panels share graphs.talk.GROUPS
TALK_CONDS = {
    "transcript_inline": "transcript_inline",
    "olens_prompt_inline": "olens_inline",
    "olens_prompt_placebo_inline": "olens_placebo_inline",
}


def talk_bars(task: str, runs: list[str] | None = None, reps: int = 2000) -> tuple[dict, float, int]:
    """condition -> (value, 95% CI, stars), chance, cases: AUROC of P(knew) (task "knew") or % right on the letter ("letter"),
    every draw one prediction, CIs a bootstrap over cases, stars a one-sided paired bootstrap (the same resamples) that the
    condition beats transcript-only. `runs`: this eval's (default) or the legacy cohort's Haiku rerun (LEGACY_HAIKU[task],
    whose conditions already carry the legacy names)."""
    key = "p" if task == "knew" else "correct"
    by: dict[str, dict[str, list[float]]] = {}
    cls: dict[str, bool] = {}
    for r in rows(task, runs):
        c = TALK_CONDS.get(r["condition"], r["condition"] if r["condition"] in TALK_CONDS.values() else None)
        if c:
            by.setdefault(c, {}).setdefault(r["item"], []).append(float(r[key]))
            cls[r["item"]] = r["class"] == "knew"
    items = sorted(set.intersection(*(set(v) for v in by.values())))
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(items), (reps, len(items)))

    def value(c: str, ix) -> float:
        picked = [items[k] for k in ix]
        if task == "knew":
            return auroc(
                np.array([p for i in picked if cls[i] for p in by[c][i]]), np.array([p for i in picked if not cls[i] for p in by[c][i]])
            )
        return float(np.mean([v for i in picked for v in by[c][i]]))

    boots = {c: np.array([value(c, ix) for ix in idx]) for c in by}
    bars = {}
    for c in by:
        st = "" if c == "transcript_inline" else stars(float((boots[c] - boots["transcript_inline"] <= 0).mean()))
        bars[c] = (value(c, range(len(items))), np.quantile(boots[c], [0.025, 0.975]).tolist(), st)
    return bars, 0.5, len(items)
