"""All standardized probe families, both positions: python -m graphs.standard_probes.

Writes full-population MCQ transfer, hinted MCQ and essay figures to docs/img/. Intervals resample
question stems or essays, conditional on the fitted out-of-fold models. No refitting or model calls.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from cfbench.scoring import grouped_auc_ci, grouped_mean_ci, standard_essay_comparison, standard_probe_rows
from graphs.style import INK, MUTED, bar_kw, ci_bar

OUT = Path("docs/img")
SITES = ("user_end", "response_start")
SUBSETS = (("All questions", None), ("Qwen right", True), ("Qwen wrong", False))
NAMES = {"mmlu": "MMLU-Pro", "supergpqa": "SuperGPQA", "combined": "Combined"}


def legend(fig):
    fig.legend(
        [Patch(**bar_kw(f"probe_{site}")) for site in SITES],
        ["User end", "Final token"],
        loc="upper right",
        bbox_to_anchor=(0.99, 1.01),
        ncol=2,
    )


def save(fig, name):
    OUT.mkdir(exist_ok=True, parents=True)
    path = OUT / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(path.resolve())


def subset(rows, right):
    return [r for r in rows if right is None or (r["label"] == r["gold"]) == right]


def accuracy(rows):
    return tuple(100 * v for v in grouped_mean_ci([r["predicted_label"] == r["label"] for r in rows], [r["group"] for r in rows]))


def mcq():
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7), sharey=True, layout="constrained")
    for row, target in enumerate(("mmlu", "supergpqa")):
        for col, (title, right) in enumerate(SUBSETS):
            ax = axes[row, col]
            for train_index, family in enumerate(NAMES):
                for site_index, site in enumerate(SITES):
                    rows = subset(standard_probe_rows(family, site, target), right)
                    ci_bar(ax, train_index * 2.5 + site_index * 0.85, f"probe_{site}", *accuracy(rows))
            ax.set_title(f"{NAMES[target]} · {title} (n = {len(rows):,})")
            ax.set_xticks([0.425 + 2.5 * i for i in range(3)], list(NAMES.values()))
            ax.set_xlim(-0.65, 6.45)
            ax.set_ylim(0, 106)
            ax.grid(axis="x", visible=False)
            if col == 0:
                ax.set_ylabel("Accuracy (%)")
    fig.suptitle("MCQ probes: training-benchmark transfer", x=0.01, ha="left")
    legend(fig)
    fig.supxlabel(
        "Training source · 1,000 questions per outer fold; combined = 500 + 500\n"
        "Nested two-fold; all held-out questions; 95% question-stem bootstrap CIs",
        fontsize=9,
        color=MUTED,
    )
    save(fig, "standard-probes-mcq.png")


def hinted():
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.9), sharey=True, layout="constrained")
    for ax, (title, right) in zip(axes, SUBSETS, strict=True):
        for x, site in enumerate(SITES):
            rows = subset(standard_probe_rows("hinted", site), right)
            ci_bar(ax, x, f"probe_{site}", *accuracy(rows), width=0.65)
        baseline = 100 * sum(r["label"] == r["gold"] for r in rows) / len(rows)
        ax.axhline(baseline, color=MUTED, ls=":", lw=1)
        ax.set_title(f"{title} (n = {len(rows):,})")
        ax.set_xticks([0, 1], ["User end", "Final token"])
        ax.set_xlim(-0.65, 1.65)
        ax.set_ylim(0, 108)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Accuracy predicting Qwen's no-hint letter (%)")
    fig.suptitle("Hinted MCQ: read-position ablation", x=0.01, ha="left")
    fig.supxlabel(
        "Gold-letter hint after the options; no disagreement or hint-following filter\n"
        "Dotted: follow the hint (78.5%, 100%, 0%). Nested two-fold; 95% question-stem CIs.",
        fontsize=8.5,
        color=MUTED,
    )
    save(fig, "standard-probes-hinted.png")


def essays():
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    scores, reference = standard_essay_comparison()
    rows = list(reference.values())
    for x, (condition, values) in enumerate(scores.items()):
        draws = [(p, r) for k, r in reference.items() for p in values[k]]  # every draw one prediction
        m, lo, hi = grouped_auc_ci([p for p, _ in draws], [r["truth"] for _, r in draws], [r["group"] for _, r in draws])
        ci_bar(ax, x, condition, m, lo, hi, width=0.65, fmt="{:.3f}")
    baseline, _, _ = grouped_auc_ci([-r["observed"] for r in rows], [r["truth"] for r in rows], [r["group"] for r in rows])
    ax.axhline(baseline, color=MUTED, ls=":", lw=1)
    names = ["Transcript", "O-lens", "placebo", "NLA", "placebo", "J++", "placebo", "Probe:\nuser end", "Probe:\nfinal token"]
    ax.set_xticks(range(len(names)), names)
    ax.set_xlim(-0.65, len(names) - 0.35)
    ax.set_ylim(0.5, 0.95)
    ax.set_ylabel("AUROC: plain-opener grade > observed grade")
    ax.grid(axis="x", visible=False)
    ax.set_title("Essays: investigators and standardized probes\n1,800 matched cases · 204 essays", color=INK)
    fig.supxlabel(
        "Probes: nested two-fold; 899/901 training cases per fold; 95% essay-bootstrap CIs\n"
        f"Dotted: mean-reversion baseline ({baseline:.3f}). Haiku 5.5 at effort medium, AUROC over every draw.\n"
        "Placebo: another opener's reads on a different essay of the same topic.",
        fontsize=8.5,
        color=MUTED,
    )
    save(fig, "standard-probes-essays.png")


if __name__ == "__main__":
    mcq()
    hinted()
    essays()
