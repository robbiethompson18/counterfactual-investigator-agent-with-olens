"""chive_inline: AUROC on CHIVE's own yes/no claims, Haiku with Qwen's reply (transcript) vs without it (edit only), beside
CHIVE's Opus 4.8 transcript-only and resampling arms on the same claims; one panel per target, plus the paired transcript −
edit-only difference per target (cfbench/evals/chive_inline, docs/chive-inline.md).

    uv run python -m graphs.chive_inline [--run data/runs/cfbench/2026-10-31_chive-haiku]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cfbench.evals.chive_inline.score import auroc
from chive_critique.data import ADAM_LABELS, ADAM_RUNS, adam
from graphs.style import MUTED, ci_bar, diff_rows

BARS = ["no_transcript_inline", "transcript_inline", "chive_opus_plain", "chive_opus_cf"]
OPUS = {"chive_opus_plain": "plain", "chive_opus_cf": "cf"}
BOOT = 1000


def per_claim(run: Path) -> dict[str, dict[str, tuple[int, dict[str, list[float]]]]]:
    """version -> claim -> (truth, condition -> its p's): Haiku's draws and CHIVE's one Opus answer per arm."""
    out: dict = defaultdict(lambda: defaultdict(lambda: [0, defaultdict(list)]))
    for r in map(json.loads, (run / "predictions.jsonl").read_text().splitlines()):
        if r.get("p") is not None:
            out[r["version"]][r["item"]][0] = r["k_cf"]
            out[r["version"]][r["item"]][1][r["condition"]].append(r["p"])
    for version, claims in out.items():
        for c, arm in OPUS.items():
            for r in adam(f"{version}/binary_evals/{arm}_claude-opus-4-8_high_conc3.jsonl"):
                if r["claim_id"] in claims and r["prob_true"] not in (None, "None"):
                    claims[r["claim_id"]][1][c].append(float(r["prob_true"]))
    return out


def aurocs(claims: dict, ids: list[str]) -> list[float]:
    """AUROC per bar over every prediction (each draw one, as each CHIVE row is) of the claims `ids`."""
    return [
        auroc(np.array([claims[i][0] for i in ids for _ in claims[i][1][c]]), np.array([p for i in ids for p in claims[i][1][c]]))
        for c in BARS
    ]


def main(run: Path, out: Path, out_diff: Path):
    data, rng = per_claim(run), np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(ADAM_RUNS), figsize=(3.3 * len(ADAM_RUNS), 3.6), sharey=True, layout="constrained")
    rows = []
    for ax, version in zip(axes, sorted(ADAM_RUNS, key=lambda v: ADAM_LABELS[v]), strict=True):
        claims = data[version]
        ids = list(claims)
        point = aurocs(claims, ids)
        boots = np.array([aurocs(claims, [ids[i] for i in rng.integers(len(ids), size=len(ids))]) for _ in range(BOOT)])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        for x, (c, v, l, h) in enumerate(zip(BARS, point, lo, hi)):
            ci_bar(ax, x, c, v, l, h, fmt="{:.3f}")
        d = boots[:, 1] - boots[:, 0]
        rows.append(
            ("transcript_inline", "no_transcript_inline", point[1] - point[0], *np.percentile(d, [2.5, 97.5]), ADAM_LABELS[version])
        )
        ax.set_xticks(range(len(BARS)), ["Haiku\nedit only", "Haiku\ntranscript", "Opus\ntranscript", "Opus\nresampling"], fontsize=8)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{ADAM_LABELS[version]}, {len(ids)} claims", fontsize=9.5)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_ylabel("AUROC of P(claim true)")
    fig.suptitle("CHIVE's yes/no claims: does seeing the model's reply help?", x=0.01, ha="left")
    fig.text(0.01, -0.04, "Haiku 5.5, effort low, 3 draws per claim; edit only = the transcript prompt without the model's reply. "
             "Opus bars: CHIVE's published Opus 4.8 arms, without and with resampling the target. 95% bootstrap CIs over claims.",
             fontsize=7.5, color=MUTED)  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(out)

    fig, ax = plt.subplots(figsize=(6.2, 0.35 + 0.34 * len(rows)))
    diff_rows(ax, rows, value_fmt="{:+.3f}")
    ax.set_xlim(-0.03, 0.06)
    ax.set_xlabel("Haiku AUROC, transcript − edit only, paired by claim (95% bootstrap CI; filled = excludes 0)", fontsize=8.5)
    fig.savefig(out_diff, bbox_inches="tight")
    print(out_diff)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=Path("data/runs/cfbench/2026-10-31_chive-haiku"))
    p.add_argument("--out", type=Path, default=Path("docs/img/chive-inline.png"))
    p.add_argument("--out-diff", type=Path, default=Path("docs/img/chive-inline-diff.png"))
    a = p.parse_args()
    main(a.run, a.out, a.out_diff)
