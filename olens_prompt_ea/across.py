# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=17", "numpy>=1.26", "scipy>=1.13", "statsmodels>=0.14"]
# ///
"""Across tasks: do prompts whose prompt-side O-lens reads contain eval words get more VEA in their rollouts?

Unit = unique prompt (and condition), pooled across every setting. x = lens score of the prompt: any strict eval-word hit
in its prompt-side reads (binary), and the share of prompt positions with a strict / broad hit. Control = the same
regexes on the raw prompt text (hits per 1k chars), so "the lens adds something" means an effect after adjusting for the
prompt itself containing eval words. y = VEA of each episode. Label sources differ (monitor / SAD answer vs regex on the
rollout), so everything is shown pooled and for the monitor-labelled settings only.

Needs kwonly.py's kwonly_episodes.parquet.   uv run --script olens_prompt_ea/across.py RUN -> RUN/across.md, across.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from analyze import boot
from keywords import BROAD, STRICT
from scipy.stats import spearmanr


def logit(d: pd.DataFrame, xs: list[str]) -> dict:
    """Episode-level logistic regression, standard errors clustered by prompt. Predictors z-scored (per-SD odds ratios)."""
    X = d[xs].astype(float)
    X = (X - X.mean()) / X.std().replace(0, 1)
    m = sm.GLM(d.vea.astype(float), sm.add_constant(X), family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(d.ph)[0]}
    )
    ci = m.conf_int()
    return {
        x: {
            "or_per_sd": float(np.exp(m.params[x])),
            "lo": float(np.exp(ci.loc[x, 0])),
            "hi": float(np.exp(ci.loc[x, 1])),
            "p": float(m.pvalues[x]),
        }
        for x in xs
    }


def section(name: str, e: pd.DataFrame) -> tuple[dict, list[str]]:
    r: dict = {"name": name, "episodes": len(e), "prompts": e.ph.nunique(), "settings": e.setting.nunique(), "vea": e.vea.mean()}
    p = e.groupby("ph").agg(vea=("vea", "mean"), n=("tid", "size"), hit=("lens_any", "first"))
    hit, no = p[p.hit], p[~p.hit]
    r["prompts_hit"], r["vea_hit_eps"], r["vea_nohit_eps"] = int(p.hit.sum()), e.vea[e.lens_any].mean(), e.vea[~e.lens_any].mean()
    r["vea_hit_prompts"], r["vea_nohit_prompts"] = hit.vea.mean(), no.vea.mean()
    for c in ("lens_strict", "lens_broad", "lens_end", "ptxt_strict"):
        r[f"auc_{c}"], r[f"auc_{c}_lo"], r[f"auc_{c}_hi"], _ = boot(e.assign(cond="all"), c, within=False)
    c = e.groupby(["setting", "cond"]).agg(
        vea=("vea", "mean"), n=("tid", "size"), lens=("lens_strict", "mean"), ptxt=("ptxt_strict", "mean")
    )
    c = c[c.n >= 5]
    r["conds"] = len(c)
    for x in ("lens", "ptxt"):
        r[f"rho_cond_{x}"], r[f"rho_cond_{x}_p"] = spearmanr(c[x], c.vea)
    s = e.groupby("setting").agg(vea=("vea", "mean"), lens=("lens_strict", "mean"), ptxt=("ptxt_strict", "mean"))
    if len(s) >= 4:
        r["rho_setting_lens"], r["rho_setting_lens_p"] = spearmanr(s.lens, s.vea)
    r["logit_lens"] = logit(e, ["lens_strict"])
    r["logit_adj"] = logit(e, ["lens_strict", "ptxt_strict"])
    r["logit_any_adj"] = logit(e.assign(lens_any=e.lens_any.astype(float)), ["lens_any", "ptxt_strict"])
    r["logit_end_adj"] = logit(e, ["lens_end", "ptxt_strict"])

    def f(x: dict) -> str:
        return f"{x['or_per_sd']:.2f} [{x['lo']:.2f}, {x['hi']:.2f}], p = {x['p']:.2g}"

    L = [
        f"**{name}**: {r['episodes']} episodes, {r['prompts']} prompts, {r['settings']} settings, VEA {r['vea']:.2f}",
        "",
        (f"- Prompts whose prompt-side reads contain a strict eval word: {r['prompts_hit']}/{r['prompts']}. VEA rate, episodes: "
        f"{r['vea_hit_eps']:.2f} (hit) vs {r['vea_nohit_eps']:.2f} (no hit); mean over prompts: {r['vea_hit_prompts']:.2f} vs "
        f"{r['vea_nohit_prompts']:.2f}."),
        (f"- Pooled AUROC vs episode VEA (prompt bootstrap): lens strict share {r['auc_lens_strict']:.2f} "
        f"[{r['auc_lens_strict_lo']:.2f}, {r['auc_lens_strict_hi']:.2f}], lens broad share {r['auc_lens_broad']:.2f} "
        f"[{r['auc_lens_broad_lo']:.2f}, {r['auc_lens_broad_hi']:.2f}], lens end of user turn {r['auc_lens_end']:.2f} "
        f"[{r['auc_lens_end_lo']:.2f}, {r['auc_lens_end_hi']:.2f}]; raw prompt text (control) {r['auc_ptxt_strict']:.2f} "
        f"[{r['auc_ptxt_strict_lo']:.2f}, {r['auc_ptxt_strict_hi']:.2f}]."),
        (f"- Logistic, episode level, prompt-clustered SEs, OR per SD: lens alone {f(r['logit_lens']['lens_strict'])}; "
        f"lens adjusted for prompt text {f(r['logit_adj']['lens_strict'])} (prompt text {f(r['logit_adj']['ptxt_strict'])}); "
        f"any lens hit (binary) adjusted {f(r['logit_any_adj']['lens_any'])}; end-of-user-turn lens "
        f"adjusted {f(r['logit_end_adj']['lens_end'])}."),
        f"- Spearman over {r['conds']} conditions (≥ 5 eps): lens {r['rho_cond_lens']:+.2f} (p = {r['rho_cond_lens_p']:.2g}), prompt "
        f"text {r['rho_cond_ptxt']:+.2f} (p = {r['rho_cond_ptxt_p']:.2g})"
        + (f"; over settings: lens {r['rho_setting_lens']:+.2f} (p = {r['rho_setting_lens_p']:.2g})." if "rho_setting_lens" in r else "."),
        "",
    ]  # fmt: skip
    return r, L


def main() -> None:
    run = Path(sys.argv[1])
    e = pd.read_parquet(run / "kwonly_episodes.parquet")
    kw = pd.read_parquet(run / "kw_scores.parquet")
    pt = pd.read_parquet(run / "prompts.parquet", columns=["ph", "prompt_text"]).set_index("ph").prompt_text
    e["ptxt_strict"] = e.ph.map(pt.map(lambda t: len(STRICT.findall(t)) / max(len(t), 1) * 1000))
    e["ptxt_broad"] = e.ph.map(pt.map(lambda t: len(BROAD.findall(t)) / max(len(t), 1) * 1000))
    e["lens_strict"] = e.ph.map(kw.kw_strict_hit_posfrac)
    e["lens_broad"] = e.ph.map(kw.kw_broad_hit_posfrac)
    e["lens_any"] = e.ph.map(kw.kw_strict_hit_count > 0).astype(bool)
    # density-robust variant: only the last 3 prompt-side positions (end of the user turn), which every setting has read;
    # dense-read prompts otherwise dilute the share with hundreds of mid-email positions
    reads = pd.read_parquet(run / "reads.parquet", columns=["ph", "pos", "text"])
    last = reads[["ph", "pos"]].drop_duplicates().sort_values("pos").groupby("ph").tail(3)
    rr = reads.merge(last, on=["ph", "pos"])
    rr["hit"] = rr.text.fillna("").str.contains(STRICT)
    e["lens_end"] = e.ph.map(rr.groupby(["ph", "pos"]).hit.any().groupby("ph").mean())
    e["labelled"] = e.monitor.notna()
    out, L = {}, ["# Across tasks: prompt-side lens eval words vs VEA rate", ""]
    per = []
    groups = {
        "All settings (monitor + regex VEA)": e,
        "All settings except donation bet (24,800 eps, 0.3% VEA)": e[e.setting != "donation_bet"],
        "Monitor / answer-labelled settings only": e[e.labelled],
        "Regex-labelled settings only": e[~e.labelled],
    }
    for name, d in groups.items():
        r, lines = section(name, d)
        out[name] = r
        L += lines
    L += ["Per setting (means over episodes):", ""]
    L.append(
        "| setting | VEA source | eps | VEA | lens strict share | lens broad share | lens end of user turn | prompt text strict /1k chars |"
    )
    L.append("|" + "---|" * 8)
    for st, d in e.groupby("setting"):
        per.append(
            {
                "setting": st, "src": "label" if d.labelled.all() else "regex", "n": len(d), "vea": d.vea.mean(),
                "lens_strict": d.lens_strict.mean(), "lens_broad": d.lens_broad.mean(), "lens_end": d.lens_end.mean(),
                "ptxt": d.ptxt_strict.mean(),
            }
        )  # fmt: skip
    for p in sorted(per, key=lambda p: p["vea"]):
        cells = [p["setting"], p["src"], str(p["n"])] + [f"{p[k]:.3f}" for k in ("vea", "lens_strict", "lens_broad", "lens_end", "ptxt")]
        L.append("| " + " | ".join(cells) + " |")
    out["per_setting"] = per
    md = "\n".join(L) + "\n"
    (run / "across.md").write_text(md)
    (run / "across.json").write_text(json.dumps(out, indent=1, default=float))
    print(md)


if __name__ == "__main__":
    main()
