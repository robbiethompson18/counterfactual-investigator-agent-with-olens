"""chive_agam_inline: AUROC on CHIVE-style claims over Agam's Qwen3.6-27B incidents (median-length half), Haiku with edit only,
transcript, and transcript plus O-lens / NLA / J++ reads, each beside its placebo; plus paired differences by claim
(cfbench/evals/chive_agam_inline, docs/chive-inline.md).

    uv run python -m graphs.chive_agam_inline --runs data/runs/cfbench/<nolens run> data/runs/cfbench/<lens run>
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cfbench.evals.chive_agam_inline.score import PAIRS, table
from graphs.style import MUTED, OLENS, ci_bar, diff_rows, positions

GROUPS = [
    ("edit only", ["no_transcript_inline"]),
    ("transcript", ["transcript_inline"]),
    (OLENS, ["olens_inline", "olens_placebo_inline"]),
    ("NLA", ["nla_inline", "nla_placebo_inline"]),
    ("J++", ["jlens_inline", "jlens_placebo_inline"]),
]


def main(runs: list[Path], out: Path, out_diff: Path):
    conds, point, boots, n = table(runs)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    xs, ticks = positions(conds, GROUPS)
    fig, ax = plt.subplots(figsize=(6.4, 3.6), layout="constrained")
    for c, x in xs.items():
        k = conds.index(c)
        ci_bar(ax, x, c, point[c], lo[k], hi[k], fmt="{:.3f}")
    ax.set_xticks(ticks, [name for name, cs in GROUPS if any(c in xs for c in cs)])
    ax.set_xlim(-0.6, ticks[-1] + 1.0)
    ax.set_ylim(0.5, min(1.0, float(hi.max()) + 0.05))
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("AUROC of P(claim true)")
    ax.set_title(f"CHIVE-style claims on Agam's Qwen3.6-27B incidents, {n} claims")
    fig.text(0.995, 0.97, "hatched = placebo", ha="right", fontsize=8, color=MUTED)
    note = ("Haiku 5.5, effort medium, 3 draws per claim. Reads of the original conversation and reply, every punctuation and "
            "end-of-turn token plus the ten tokens before the reply,\nlayers 32-60. 95% bootstrap CIs over claims.")  # fmt: skip
    fig.text(0.01, -0.05, note, fontsize=7, color=MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(out)

    rows = []
    for a, b in PAIRS:
        if a in conds and b in conds:
            d = boots[:, conds.index(a)] - boots[:, conds.index(b)]
            rows.append((a, b, point[a] - point[b], *np.percentile(d, [2.5, 97.5])))
    fig, ax = plt.subplots(figsize=(6.2, 0.35 + 0.34 * len(rows)))
    diff_rows(ax, rows, value_fmt="{:+.3f}")
    ax.set_xlim(-0.04, 0.06)
    ax.set_xlabel("AUROC difference, paired by claim (95% bootstrap CI; filled = excludes 0)", fontsize=8.5)
    fig.savefig(out_diff, bbox_inches="tight")
    print(out_diff)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, default=Path("docs/img/chive-agam-inline.png"))
    p.add_argument("--out-diff", type=Path, default=Path("docs/img/chive-agam-inline-diff.png"))
    a = p.parse_args()
    main(a.runs, a.out, a.out_diff)
