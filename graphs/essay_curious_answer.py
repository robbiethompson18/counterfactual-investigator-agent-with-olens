"""Charts for the essay_curious answer task (docs/envs/essay_curious.md): % of predictor draws naming the letter grade Qwen
gives with no scenario, per condition, with 95% case-bootstrap CIs and the "edit does nothing" null; plus a grid of the
fraction of correct draws per case (essay x scenario arm) and condition.

    uv run python -m graphs.essay_curious_answer RUN_DIR [RUN_DIR ...] [--out docs/img/essay-curious]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cfbench.answer import load, modal
from cfbench.core import canon_condition
from graphs.style import INK, MUTED, bar_kw, marker_kw

CONDS = ["no_transcript", "transcript_agentic", "olens_agentic", "olens_placebo_agentic"]
LABEL = {
    "no_transcript": "No transcript",
    "transcript_agentic": "Transcript-only",
    "olens_agentic": "O-lens",
    "olens_placebo_agentic": "O-lens placebo",
}
N_BOOT = 4000


def per_case(runs: list[Path]) -> tuple[list[tuple], dict[str, dict[tuple, float]], dict[tuple, dict]]:
    """Cases (item, base arm), condition -> case -> fraction of draws correct, and case -> the case fields."""
    groups, _ = load([r / "predictions.jsonl" for r in runs])
    by: dict[str, dict[tuple, float]] = {}
    fields: dict[tuple, dict] = {}
    for (cond, _), group in groups.items():
        for (env, _, base_arm, _), c in group.items():
            key = (env.rsplit("@", 1)[1], base_arm)
            by.setdefault(canon_condition(cond), {})[key] = np.mean([a == c["modal_cf"] for a in c["canon"]])
            fields[key] = c
    return sorted(fields), by, fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/img/essay-curious"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    keys, by, fields = per_case(a.runs)
    conds = [c for c in CONDS if c in by]
    null = np.mean([modal(fields[k]["answers_base"]) == fields[k]["modal_cf"] for k in keys])
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, c in enumerate(conds):
        v = np.array([by[c][k] for k in keys])
        boot = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ax.bar(i, v.mean() * 100, width=0.65, **bar_kw(c))
        ax.errorbar(i, v.mean() * 100, yerr=[[(v.mean() - lo) * 100], [(hi - v.mean()) * 100]], color=MUTED, capsize=4, lw=1)
        ax.text(i, hi * 100 + 2, f"{v.mean() * 100:.1f}%", ha="center", color=INK)
        print(f"{LABEL[c]:16} {v.mean() * 100:5.1f}%  [{lo * 100:.0f}, {hi * 100:.0f}]  ({len(v)} cases)")
    ax.axhline(null * 100, color=MUTED, ls="--", lw=1)
    ax.text(len(conds) - 0.5, null * 100 + 1.5, f"edit does nothing {null * 100:.0f}%", ha="right", color=MUTED, fontsize=9)
    ax.set_xticks(range(len(conds)), [LABEL[c] for c in conds])
    ax.set_ylim(0, 105)
    ax.set_ylabel("draws naming the no-scenario grade (%)")
    ax.set_title(f"Predict Qwen's letter grade with the scenario removed ({len(keys)} cases)")
    fig.tight_layout()
    fig.savefig(a.out / "accuracy.png")
    print(f"edit does nothing: {null * 100:.0f}%")

    fig, ax = plt.subplots(figsize=(4.2 + 1.3 * len(conds), 0.42 * len(keys) + 1.3))
    for j, c in enumerate(conds):
        for i, k in enumerate(keys):
            ax.scatter(j, i, s=60 + 340 * by[c][k], alpha=0.25 + 0.75 * by[c][k], **marker_kw(c))
            ax.text(j + 0.22, i, f"{by[c][k] * 100:.0f}", va="center", fontsize=8, color=MUTED)

    def label(item, arm):
        c = fields[(item, arm)]
        tie = sum(v == c["k_cf"] for v in c["answers_cf"].values()) > 1  # the no-scenario arm has no single modal grade
        return f"{item}, {arm}: {modal(c['answers_base'])} → {c['modal_cf']}{' (tie)' if tie else ''}"

    labels = [label(item, arm) for item, arm in keys]
    ax.set_yticks(range(len(keys)), labels)
    ax.set_xticks(range(len(conds)), [LABEL[c].replace(" ", "\n", 1) for c in conds])
    ax.xaxis.tick_top()
    ax.set_xlim(-0.5, len(conds) - 0.3)
    ax.invert_yaxis()
    ax.grid(False)
    ax.set_title("% of draws correct, per case (scenario arm's grade → no-scenario grade)", loc="right", pad=34)
    fig.tight_layout()
    fig.savefig(a.out / "per_case.png")


if __name__ == "__main__":
    main()
