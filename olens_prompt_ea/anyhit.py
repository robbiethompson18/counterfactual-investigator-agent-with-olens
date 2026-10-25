# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.1", "pandas>=2.2", "pyarrow>=17", "numpy>=1.26", "scipy>=1.13", "matplotlib>=3.9"]
# ///
"""Any-hit version with Kylie's regex (simulation|eval|scenario, case-insensitive), per episode:
  prompt_any  >= 1 read at a prompt-side position (before the first assistant turn) matches; pooled over every episode
              with the same prompt (identical prompts were often read only once)
  gen_any     >= 1 read at a position after the first assistant turn matches (the model's own reasoning / reply / tools)
  all_any     either
for two reader sets: GLM O-lens (SFT + RL, greedy + 4 samples; what the earlier analysis used) and every reader the
8830 viewer shows (+ Qwen distilled / from-scratch, J-lens L60 / L51 top-10).

Writes RUN/anyhit_episodes.parquet, RUN/anyhit_settings.csv, the figure, and prints per-setting table + pooled AUROCs.

    uv run --script olens_prompt_ea/anyhit.py RUN --db data/olens_search/olens.duckdb --fig docs/figures/olens-prompt-ea-anyhit.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy.stats import rankdata

mpl.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from fig_settings import NAMES

RX = "simulation|eval|scenario"
RNG = np.random.default_rng(0)


def auc(y: np.ndarray, s: np.ndarray) -> float:
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def boot_ci(d: pd.DataFrame, f, b: int = 500) -> tuple[float, float]:
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
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--fig", type=Path, required=True)
    ap.add_argument(
        "--glm-filter", default="r.reader = 'glm_rl'", help="SQL filter for the main reader set (default: GLM RL lens, greedy + 4 samples)"
    )
    ap.add_argument("--tag", default="", help="suffix for the output files")
    a = ap.parse_args()
    e = pd.read_parquet(a.run / "kwonly_episodes.parquet")[["tid", "setting", "cond", "ph", "vea", "monitor"]]
    con = duckdb.connect(str(a.db), read_only=True)
    con.register("eps", e[["tid"]])
    hits = con.execute(f"""
        with pe as (select e.tid, strpos(e.rendered_text, '<|assistant|>') - 1 as pend from episodes e join eps using (tid))
        select p.tid, p.pos, (p.char_end >= 0 and p.char_end <= pe.pend) as prompt_side,
               bool_or({a.glm_filter} and regexp_matches(r.text, '{RX}', 'i')) as glm_hit,
               bool_or(regexp_matches(r.text, '{RX}', 'i')) as any_reader_hit
        from positions p join pe on p.tid = pe.tid join reads r on r.tid = p.tid and r.pos = p.pos
        group by all""").df()
    h = hits.merge(e[["tid", "ph"]], on="tid")
    out = e.copy()
    for rd in ("glm", "any_reader"):
        col = f"{rd}_hit"
        pr = h[h.prompt_side].groupby("ph")[col].any()  # pooled per prompt
        ge = h[~h.prompt_side].groupby("tid")[col].any()
        out[f"{rd}_prompt_any"] = out.ph.map(pr).fillna(False).astype(bool)
        out[f"{rd}_gen_any"] = out.tid.map(ge).fillna(False).astype(bool)
        out[f"{rd}_all_any"] = out[f"{rd}_prompt_any"] | out[f"{rd}_gen_any"]
        out[f"{rd}_prompt_share"] = out.ph.map(h[h.prompt_side].groupby(["ph", "pos"])[col].any().groupby("ph").mean())
    out["has_prompt_reads"] = out.ph.isin(h[h.prompt_side].ph)
    out.to_parquet(a.run / f"anyhit_episodes{a.tag}.parquet")

    rows = []
    for st, d in out.groupby("setting"):
        d = d.reset_index(drop=True)
        r = {"setting": st, "label": "answer" if st == "sad_oversight" else ("monitor" if d.monitor.notna().all() else "regex"),
             "n": len(d), "prompts": d.ph.nunique(), "vea": d.vea.mean()}  # fmt: skip
        for c in ("glm_prompt_any", "glm_gen_any", "glm_all_any", "any_reader_prompt_any", "any_reader_all_any"):
            r[c] = d[c].mean()
            r[c + "_lo"], r[c + "_hi"] = boot_ci(d, lambda x, c=c: x[c].mean(), 300)
        r["vea_lo"], r["vea_hi"] = boot_ci(d, lambda x: x.vea.mean(), 300)
        for c in ("glm_prompt_any", "glm_gen_any"):
            g = d[c]
            r[f"vea_if_{c}"] = d.vea[g].mean() if g.any() else np.nan
            r[f"vea_if_not_{c}"] = d.vea[~g].mean() if (~g).any() else np.nan
        rows.append(r)
    s = pd.DataFrame(rows).sort_values("vea").reset_index(drop=True)
    a.fig.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(a.fig.with_suffix(".csv"), index=False)

    print(f"regex {RX!r}; per setting (episode rates):")
    cols = ["setting", "label", "n", "vea", "glm_prompt_any", "glm_gen_any", "glm_all_any", "any_reader_prompt_any", "any_reader_all_any"]
    cols += ["vea_if_glm_prompt_any", "vea_if_not_glm_prompt_any", "vea_if_glm_gen_any", "vea_if_not_glm_gen_any"]
    print(s[cols].round(3).to_string())
    for name, d in (("all", out), ("no donation bet", out[out.setting != "donation_bet"]), ("monitor-labelled", out[out.monitor.notna()])):
        d = d.reset_index(drop=True)
        y = d.vea.to_numpy(bool)
        line = [f"{name} (n={len(d)}):"]
        for c in ("glm_prompt_any", "glm_prompt_share", "glm_gen_any", "any_reader_prompt_any"):
            v = auc(y, d[c].fillna(0).to_numpy(float))
            lo, hi = boot_ci(d, lambda x, c=c: auc(x.vea.to_numpy(bool), x[c].fillna(0).to_numpy(float)), 200)
            line.append(f"{c} AUROC {v:.2f} [{lo:.2f}, {hi:.2f}]")
        print("  " + "; ".join(line))
    sl = s[s.n >= 50]
    from scipy.stats import spearmanr

    for c in ("glm_prompt_any", "glm_gen_any", "glm_all_any"):
        rho, p = spearmanr(sl[c], sl.vea)
        print(f"  setting-level Spearman VEA vs {c}: {rho:+.2f} (p = {p:.2g}, {len(sl)} settings)")

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#8a8984", "xtick.color": "#52514e", "ytick.color": "#2b2a27"})
    fig, ax = plt.subplots(figsize=(8.4, 9.4), dpi=160)
    y = np.arange(len(s))
    hgt = 0.26
    series = (("vea", "VEA rate of rollouts", "#2a78d6"),
              ("glm_prompt_any", "Episodes with a prompt-side read matching simulation|eval|scenario", "#eb6834"),
              ("glm_gen_any", "Episodes whose generation-side O-lens reads match it", "#1baf7a"))  # fmt: skip
    for j, (k, lab, col) in enumerate(series):
        yy = y + (1 - j) * hgt
        ax.barh(yy, s[k], height=hgt - 0.03, color=col, label=lab, zorder=2)
        ax.errorbar(
            s[k], yy, xerr=[s[k] - s[k + "_lo"], s[k + "_hi"] - s[k]], fmt="none", ecolor="#3b3a37", elinewidth=0.8, capsize=1.5, zorder=3
        )
    ax.set_yticks(y, [f"{NAMES.get(st, st)}  ({lab}, n={n:,})" for st, lab, n in zip(s.setting, s.label, s.n)])
    for t, lab in zip(ax.get_yticklabels(), s.label):
        t.set_fontweight("normal" if lab == "regex" else "bold")
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of episodes (95% CI, prompts resampled)")
    ax.grid(axis="x", color="#e4e3df", zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    ax.set_title(f"VEA rate vs any O-lens read matching simulation|eval|scenario ({a.glm_filter})\n"
                 "bold = monitor / answer VEA label; normal = regex on the rollout", fontsize=10, loc="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(a.fig, facecolor="white")


if __name__ == "__main__":
    main()
