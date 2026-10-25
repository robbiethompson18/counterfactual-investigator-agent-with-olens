# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=17", "numpy>=1.26", "matplotlib>=3.9"]
# ///
"""Per-setting bars: VEA rate of the rollouts vs share of prompt-side O-lens read positions with a strict eval word, plus
the share of prompts whose raw text has one (control). 95% CIs bootstrap prompts. Sorted by VEA rate.

    uv run --script olens_prompt_ea/fig_settings.py RUN docs/figures/olens-prompt-ea-settings.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from keywords import STRICT

RNG = np.random.default_rng(0)
COLORS = {"vea": "#2a78d6", "lens": "#eb6834", "ptxt": "#1baf7a"}  # reference categorical slots 1-3
NAMES = {
    "am_main": "Agentic misalignment", "hp_am_fixed": "AISI AM fixed", "hp_mc": "AISI MC", "hp_abst": "AISI AbstentionBench",
    "am": "Agentic misalignment (ours, 15 conds + AISI fixed)",
    "sad_oversight": "SAD stages_oversight", "inspect_am": "AM (Celeste, inspect)", "instrumentaleval": "InstrumentalEval",
    "mask": "MASK", "agentharm": "AgentHarm", "ib_realistic": "IB realistic", "donation_bet": "Donation bet",
    "ec_scheming": "evalcue scheming", "ec_shutdown": "evalcue shutdown", "ec_sandbagging": "evalcue sandbagging",
    "ec_whistleblow": "evalcue whistleblow", "ec_overnight": "evalcue overnight", "ec_cue": "evalcue eval-cue envs",
    "ec_other": "evalcue other", "ec_hiring": "evalcue hiring", "ec_dumb_user": "evalcue dumb user",
    "ec_hidden_goal": "evalcue hidden goal",
}  # fmt: skip


def boot(units: list[np.ndarray], stat, b: int = 1000) -> tuple[float, float]:
    """CI of stat(concatenated arrays), resampling the prompt units."""
    vals = [stat(np.concatenate([units[i] for i in RNG.integers(0, len(units), len(units))])) for _ in range(b)]
    return tuple(np.percentile(vals, [2.5, 97.5]))


def main() -> None:
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    e = pd.read_parquet(run / "kwonly_episodes.parquet")
    pt = pd.read_parquet(run / "prompts.parquet", columns=["ph", "prompt_text"]).set_index("ph").prompt_text
    reads = pd.read_parquet(run / "reads.parquet", columns=["ph", "pos", "text"], filters=[("ph", "in", list(e.ph.unique()))])
    reads["hit"] = reads.text.fillna("").str.contains(STRICT)
    pos = reads.groupby(["ph", "pos"]).hit.any()
    pos_by_ph = {ph: g.to_numpy(float) for ph, g in pos.groupby(level=0)}
    ptxt = pt.map(lambda t: float(bool(STRICT.search(t))))
    rows = []
    for st, d in e.groupby("setting"):
        phs = d.ph.unique()
        vea_units = [g.to_numpy(float) for _, g in d.groupby("ph").vea]
        lens_units = [pos_by_ph[p] for p in phs if p in pos_by_ph]
        pt_units = [np.full(len(g), ptxt[p]) for p, g in d.groupby("ph")]
        lab = "answer" if st == "sad_oversight" else "monitor" if d.monitor.notna().all() else "regex"
        r = {"setting": st, "label": lab, "n": len(d), "prompts": len(phs)}
        for k, u in (("vea", vea_units), ("lens", lens_units), ("ptxt", pt_units)):
            r[k] = np.concatenate(u).mean()
            r[k + "_lo"], r[k + "_hi"] = boot(u, np.mean)
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("vea").reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out.with_suffix(".csv"), index=False)

    plt.rcParams.update(
        {"font.size": 9, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2a27", "xtick.color": "#52514e", "ytick.color": "#2b2a27"}
    )
    fig, ax = plt.subplots(figsize=(8.2, 9.2), dpi=160)
    y = np.arange(len(df))
    h = 0.26
    series = (("vea", "VEA rate of rollouts"), ("lens", "Prompt-side O-lens read positions with a strict eval word"),
              ("ptxt", "Prompts whose raw text has a strict eval word (control)"))  # fmt: skip
    for j, (k, lab) in enumerate(series):
        yy = y + (1 - j) * h
        ax.barh(yy, df[k], height=h - 0.03, color=COLORS[k], label=lab, zorder=2)
        ax.errorbar(
            df[k],
            yy,
            xerr=[df[k] - df[k + "_lo"], df[k + "_hi"] - df[k]],
            fmt="none",
            ecolor="#3b3a37",
            elinewidth=0.8,
            capsize=1.5,
            zorder=3,
        )
    labels = [f"{NAMES.get(s, s)}  ({lab}, n={n:,})" for s, lab, n in zip(df.setting, df.label, df.n)]
    ax.set_yticks(y, labels)
    for t, lab in zip(ax.get_yticklabels(), df.label):
        t.set_fontweight("normal" if lab == "regex" else "bold")
    ax.set_xlim(0, 1)
    ax.set_xlabel("share (95% CI, prompts resampled)")
    ax.grid(axis="x", color="#e4e3df", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_title(
        "VEA rate vs eval words in prompt-side O-lens reads, per setting (sorted by VEA)\n"
        "bold = monitor label (SAD: the model's testing/deployment answer); normal = regex on the rollout",
        fontsize=10, loc="left",
    )  # fmt: skip
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
