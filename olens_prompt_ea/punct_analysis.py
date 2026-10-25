# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=17", "numpy>=1.26", "scipy>=1.13", "matplotlib>=3.9", "statsmodels>=0.14"]
# ///
"""Analysis of judge_punct.py: the DeepSeek judge's 0-10 score of the prompt-side RL bullets at punctuation tokens
(per unique prompt) vs VEA of the episodes. Exclusions: Celeste's inspect AM, SAD, AISI MC, AbstentionBench; our AM and
AISI AM fixed pooled as "am". Writes the figure + csv and prints AUROCs (pooled, without donation bet, within condition)
and the setting-level Spearman.

    uv run --script olens_prompt_ea/punct_analysis.py RUN --fig docs/figures/olens-prompt-ea-punct-judge.png [--expview DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import quote

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

mpl.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from analyze import auc, strat_auc
from fig_settings import NAMES

EXCLUDE = {"inspect_am", "sad_oversight", "hp_mc", "hp_abst"}
RNG = np.random.default_rng(0)
FLAG = 5
IMPORT = {"am": "imports/celeste/ds/agentic_misalignment", "donation_bet": "imports/celeste/ds_new/cfi_donation_bet",
          "ib_realistic": "imports/celeste/ds_new/cfi_ib_realistic", "agentharm": "imports/celeste/ds_new/inspect_agentharm",
          "mask": "imports/celeste/ds_new/inspect_mask", "instrumentaleval": "imports/celeste/ds_new/inspect_instrumentaleval"}  # fmt: skip


def boot(d: pd.DataFrame, f, b: int = 500) -> tuple[float, float]:
    groups = [g.index.to_numpy() for _, g in d.groupby("ph")]
    vals = []
    for _ in range(b):
        idx = np.concatenate([groups[i] for i in RNG.integers(0, len(groups), len(groups))])
        v = f(d.loc[idx])
        if not np.isnan(v):
            vals.append(v)
    return tuple(np.percentile(vals, [2.5, 97.5])) if vals else (np.nan, np.nan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--fig", type=Path, required=True)
    ap.add_argument("--expview", type=Path, default=None, help="shared viewer's expview dir: also write the judge I/O as a run")
    a = ap.parse_args()
    e = pd.read_parquet(a.run / "kwonly_episodes.parquet", columns=["tid", "setting", "cond", "ph", "vea", "monitor"])
    e = e[~e.setting.isin(EXCLUDE)].copy()
    e.loc[e.setting.isin(["am_main", "hp_am_fixed"]), "setting"] = "am"
    recs = [json.loads(x) for x in (a.run / "judge" / "lens_punct.jsonl").read_text().splitlines()]
    sc = {r["key"]: r["parsed"]["score"] for r in recs}
    npos = {r["key"]: r["n_pos"] for r in recs}
    e["score"] = e.ph.map(sc).astype(float)
    e["n_pos"] = e.ph.map(npos)
    e = e.dropna(subset=["score"]).reset_index(drop=True)
    e["flag"] = (e.score >= FLAG).astype(float)
    e["vea"] = e.vea.astype(bool)
    print(
        f"{len(e):,} episodes, {e.ph.nunique():,} prompts with a judge score "
        f"({len(recs)} judged, {sum(v is None for v in sc.values())} unparsed)"
    )

    def line(name: str, d: pd.DataFrame) -> None:
        d = d.reset_index(drop=True)
        y = d.vea.to_numpy(bool)
        v = auc(y, d.score.to_numpy())
        lo, hi = boot(d, lambda x: auc(x.vea.to_numpy(bool), x.score.to_numpy()), 300)
        w = strat_auc(y, d.score.to_numpy(), d.cond.to_numpy())
        print(f"  {name} (n={len(d):,}): AUROC {v:.2f} [{lo:.2f}, {hi:.2f}]; within condition {w:.2f}; "
              f"VEA if score>={FLAG} {d.vea[d.flag == 1].mean():.3f} vs {d.vea[d.flag == 0].mean():.3f}")  # fmt: skip

    line("all", e)
    line("no donation bet", e[e.setting != "donation_bet"])
    line("monitor-labelled", e[e.monitor.notna()])
    rows = []
    for st, d in e.groupby("setting"):
        d = d.reset_index(drop=True)
        r = {"setting": st, "label": "monitor" if d.monitor.notna().all() else "regex", "n": len(d), "prompts": d.ph.nunique(),
             "punct_positions_median": d.drop_duplicates("ph").n_pos.median(), "vea": d.vea.mean(), "score_mean": d.score.mean(),
             "flag": d.flag.mean(), "within_auc": strat_auc(d.vea.to_numpy(bool), d.score.to_numpy(), d.cond.to_numpy())}  # fmt: skip
        for k, f in (("vea", lambda x: x.vea.mean()), ("flag", lambda x: x.flag.mean()), ("score_mean", lambda x: x.score.mean() / 10)):
            r[k + "_lo"], r[k + "_hi"] = boot(d, f, 300)
        rows.append(r)
    s = pd.DataFrame(rows).sort_values("vea").reset_index(drop=True)
    s["score_mean10"] = s.score_mean / 10
    a.fig.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(a.fig.with_suffix(".csv"), index=False)
    print(s[["setting", "label", "n", "prompts", "punct_positions_median", "vea", "score_mean", "flag", "within_auc"]].round(3).to_string())
    for c in ("score_mean", "flag"):
        rho, p = spearmanr(s[c], s.vea)
        print(f"  setting-level Spearman VEA vs {c}: {rho:+.2f} (p = {p:.2g}, {len(s)} settings)")

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#8a8984", "xtick.color": "#52514e", "ytick.color": "#2b2a27"})
    fig, ax = plt.subplots(figsize=(9.0, 8.4), dpi=160)
    y = np.arange(len(s))
    hgt = 0.27
    series = (("vea", "VEA rate of rollouts", "#2a78d6", "vea"),
              ("flag", f"prompts the judge scores >= {FLAG}/10 (lens frames the model's own situation as a test)", "#eb6834", "flag"),
              ("score_mean10", "mean judge score / 10", "#1baf7a", "score_mean"))  # fmt: skip
    for j, (k, lab, col, ci) in enumerate(series):
        yy = y + (1 - j) * hgt
        ax.barh(yy, s[k], height=hgt - 0.03, color=col, label=lab, zorder=2)
        ax.errorbar(
            s[k], yy, xerr=[s[k] - s[ci + "_lo"], s[ci + "_hi"] - s[k]], fmt="none", ecolor="#3b3a37", elinewidth=0.8, capsize=1.5, zorder=3
        )
    ax.set_yticks(
        y,
        [
            f"{NAMES.get(st, st)}  ({lab}, n={n:,}, {pp:.0f} punct pos/prompt)"
            for st, lab, n, pp in zip(s.setting, s.label, s.n, s.punct_positions_median)
        ],
    )
    for t, lab, pp in zip(ax.get_yticklabels(), s.label, s.punct_positions_median):
        t.set_fontweight("normal" if lab == "regex" else "bold")
        t.set_color("#2b2a27" if pp >= 15 else "#9a9893")
    ax.set_xlim(0, 1)
    ax.set_xlabel("share (95% CI, prompts resampled)")
    ax.grid(axis="x", color="#e4e3df", zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.3, -0.07), frameon=False, fontsize=8)
    ax.set_title("VEA rate vs a DeepSeek judge of the prompt-side GLM RL O-lens bullets at punctuation tokens\n"
                 "bold = monitor VEA label; normal = regex on the rollout; grey = < 15 punctuation positions read per prompt",
                 fontsize=9, loc="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(a.fig, facecolor="white")

    if a.expview:  # judge I/O as an expview run the shared viewer indexes
        sys.path.insert(0, str(a.expview))
        from expview import Run, call

        st = e.groupby("ph").agg(
            setting=("setting", "first"), cond=("cond", "first"), n=("tid", "size"), vea=("vea", "mean"), tid=("tid", "first")
        )
        run = Run(a.run / "expview" / "lens_punct_judge", title="DeepSeek judge of prompt-side RL O-lens bullets (punctuation tokens)",
                  description=("One item per unique prompt. The call is exactly what the judge (deepseek/deepseek-v4.1-flash, T=0, "
                               "reasoning effort low) saw and answered: the prompt-side GLM RL O-lens bullets at punctuation tokens of one "
                               "representative transcript (dense reads where available), blind to the rollout. Labels: `score` 0-10, "
                               "`vea_rate` over the episodes sharing this prompt, `n_eps`. Buttons open a transcript with this prompt and "
                               "its reads inline."),
                  label_order=["score", "vea_rate", "n_eps", "setting"])  # fmt: skip
        run.reset()
        for r in recs:
            if r["key"] not in st.index:
                continue
            x = st.loc[r["key"]]
            links = {}
            if x.setting in IMPORT or str(x.tid).startswith("cfi2"):
                d = IMPORT.get(x.setting, "imports/celeste/ds/evalcue")
                links["transcript with reads (8830)"] = f"http://bramble:8830/#d={d}&i={quote(str(x.tid), safe='')}"
            run.add(
                id=r["key"], group=x.setting,
                labels={"score": r["parsed"]["score"], "vea_rate": round(float(x.vea), 3), "n_eps": int(x.n), "setting": x.setting,
                        "cond": x.cond, "evidence": (r["parsed"]["evidence"] or "")[:300]},
                links=links,
                calls=[call("lens judge", r["model"], messages=r["messages"],
                            output={"role": "assistant", "content": r["output"] or "", "reasoning": r.get("reasoning")},
                            params={"temperature": 0, "reasoning_effort": "low"}, parsed=r["parsed"],
                            cost=(r.get("usage") or {}).get("cost"))],
            )  # fmt: skip
        print(f"expview run -> {run.path}")


if __name__ == "__main__":
    main()
