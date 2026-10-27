# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=17", "numpy>=1.26", "scipy>=1.13", "statsmodels>=0.14"]
# ///
"""Does prompt-side O-lens eval awareness predict verbalised eval awareness (VEA) in the rollout?

Per episode: VEA label (existing or judge.py's), and three prompt-level scores (identical for episodes sharing a prompt):
  lens    Sonnet 5.5 0-10 on the prompt-side O-lens bullets          (judge/lens.jsonl)
  prompt  Sonnet 5.5 0-10 on the raw prompt alone, no lens (control)  (judge/prompt.jsonl)
  kw      keyword score: share of prompt positions with a strict eval-word hit in any read (kw_scores.parquet),
          plus the "novel" variant (word absent from the position's own context)

Metrics per setting: pooled AUROC, within-condition AUROC (only pairs from the same condition are compared, so a score
that just tells conditions apart gets 0.5), bootstrap CIs resampling prompts (prompts are the unit: episodes sharing a
prompt share every score), Fisher / Mantel-Haenszel odds ratio for lens >= FLAG, and across conditions the Spearman
correlation of mean score vs VEA rate.

    uv run --script olens_prompt_ea/analyze.py RUN   -> RUN/results.json, RUN/results.md, RUN/examples.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, rankdata, spearmanr
from statsmodels.stats.contingency_tables import StratifiedTable

FLAG = 5  # lens judge score counted as "lens flags eval awareness"
SCORES = {
    "lens": "lens judge", "prompt": "prompt-only judge", "prompt_task": "prompt-only judge, task-matched (SAD only)",
    "kw_strict": "keyword (strict)", "kw_strict_novel": "keyword novel",
}  # fmt: skip
VIEW = "http://bramble:8821/#ep={tid}&pos={pos}&q={q}"
Q = "test|eval|synthetic|simulat|scenario|fiction|hypothetical|honeypot|benchmark"
B = 1000
RNG = np.random.default_rng(0)


def auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUROC (ties count half), rank-based so it scales to tens of thousands of episodes."""
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    r = rankdata(s)
    return (r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def boot_ci(d: pd.DataFrame, f, b: int = 500) -> tuple[float, float]:
    """95% CI of f(d) resampling the prompts (`ph` groups) with replacement; NaN draws are dropped."""
    groups = [g.index.to_numpy() for _, g in d.groupby("ph")]
    vals = []
    for _ in range(b):
        idx = np.concatenate([groups[i] for i in RNG.integers(0, len(groups), len(groups))])
        v = f(d.loc[idx])
        if not np.isnan(v):
            vals.append(v)
    return tuple(np.percentile(vals, [2.5, 97.5])) if vals else (np.nan, np.nan)


def strat_auc(y: np.ndarray, s: np.ndarray, g: np.ndarray) -> float:
    """AUROC over (VEA, non-VEA) pairs from the same condition only."""
    num = den = 0.0
    for c in np.unique(g):
        m = g == c
        pos, neg = s[m & y], s[m & ~y]
        if len(pos) and len(neg):
            num += (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
            den += len(pos) * len(neg)
    return num / den if den else np.nan


def boot(df: pd.DataFrame, col: str, within: bool) -> tuple[float, float, float, int]:
    """Point estimate + 95% CI resampling prompts (within condition when `within`). Returns (est, lo, hi, n_pairs)."""
    d = df.dropna(subset=[col])
    y, s, g = d.vea.to_numpy(bool), d[col].to_numpy(float), d.cond.to_numpy()
    f = (lambda y, s, g: strat_auc(y, s, g)) if within else (lambda y, s, g: auc(y, s))
    est = f(y, s, g)
    if np.isnan(est):
        return np.nan, np.nan, np.nan, 0
    npairs = sum(((g == c) & y).sum() * ((g == c) & ~y).sum() for c in np.unique(g)) if within else y.sum() * (~y).sum()
    groups = {k: v.to_numpy() for k, v in d.reset_index(drop=True).groupby("ph").groups.items()}
    keys = np.array(list(groups))
    vals = []
    for _ in range(B):
        idx = np.concatenate([groups[k] for k in RNG.choice(keys, len(keys))])
        v = f(y[idx], s[idx], g[idx])
        if not np.isnan(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return est, lo, hi, int(npairs)


def load_judge(run: Path, task: str) -> pd.DataFrame:
    p = run / "judge" / f"{task}.jsonl"
    if not p.exists():
        return pd.DataFrame(columns=["key"])
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    return pd.DataFrame(
        [
            {"key": r["key"], **r["parsed"], "judge_model": r.get("model"), "messages": r["messages"], "output": r.get("output")}
            for r in rows
        ]
    ).drop_duplicates("key", keep="last")


def evidence_pos(messages: list[dict], evidence: str | None) -> int | None:
    """Map the judge's quoted "[k]" (or "pos N") back to the token position it saw."""
    if not evidence:
        return None
    m = re.search(r"pos(?:ition)?\s+(\d+)", evidence)
    if m:
        return int(m.group(1))
    m = re.search(r"\[(\d+)\]", evidence)
    if not m:
        return None
    m2 = re.search(rf"^\[{m.group(1)}\] pos (\d+),", messages[0]["content"], re.MULTILINE)
    return int(m2.group(1)) if m2 else None


def build(run: Path) -> pd.DataFrame:
    e = pd.read_parquet(run / "episodes.parquet").drop(columns=["gen_text"])
    e["vea"], e["vea_source"] = e.aware.astype("boolean"), np.where(e.aware.notna(), "existing monitor (honeypot run)", None)
    if (run / "am_labels.parquet").exists():
        am = pd.read_parquet(run / "am_labels.parquet").set_index("tid")
        m = e.tid.isin(am.index)
        e.loc[m, "vea"] = e.loc[m, "tid"].map(am.vea).astype("boolean")
        e.loc[m, "vea_source"] = "Sonnet 4.5 majority of 3 (AM viewer)"
    v = load_judge(run, "vea")
    if len(v):
        vm = dict(zip(v.key, v.aware))
        m = e.tid.isin(vm) & e.vea.isna()
        e.loc[m, "vea"] = e.loc[m, "tid"].map(vm).astype("boolean")
        e.loc[m, "vea_source"] = "Sonnet 5.5 monitor v2 (this run)"
    sad = load_judge(run, "sad")
    if len(sad):
        sm = dict(zip(sad.key, sad.testing))
        m = e.tid.isin(sm)
        e.loc[m, "vea"] = e.loc[m, "tid"].map(sm).astype("boolean")
        e.loc[m, "vea_source"] = "SAD answer = testing (Sonnet 5.5 parse)"
    for task in ("lens", "prompt"):
        j = load_judge(run, task)
        e[task] = e.ph.map(dict(zip(j.key, j.score))).astype(float) if len(j) else np.nan
    sp = load_judge(run, "sad_prompt")
    e["prompt_task"] = e.ph.map(dict(zip(sp.key, sp.score))).astype(float) if len(sp) else np.nan
    kw = pd.read_parquet(run / "kw_scores.parquet")
    e["kw_strict"] = e.ph.map(kw.kw_strict_hit_posfrac).astype(float)
    e["kw_strict_novel"] = e.ph.map(kw.kw_strict_novel_posfrac).astype(float)
    e["kw_broad"] = e.ph.map(kw.kw_broad_hit_posfrac).astype(float)
    e["kw_strict_count"] = e.ph.map(kw.kw_strict_hit_count).astype(float)
    e["kw_strict_max"] = e.ph.map(kw.kw_strict_hit_max).astype(float)
    return e


def setting_rows(e: pd.DataFrame) -> list[dict]:
    out = []
    for st, d in e[e.vea.notna()].groupby("setting"):
        d = d.copy()
        d["vea"] = d.vea.astype(bool)
        r = {
            "setting": st,
            "n": len(d),
            "prompts": d.ph.nunique(),
            "conds": d.cond.nunique(),
            "conds_multi_prompt": int((d.groupby("cond").ph.nunique() > 1).sum()),
            "vea_rate": d.vea.mean(),
            "vea_source": "; ".join(sorted(d.vea_source.dropna().unique())),
            "lens_mean": d.lens.mean(),
            "prompt_mean": d.prompt.mean(),
            "kw_mean": d.kw_strict.mean(),
            "kw_any": (d.kw_strict_count > 0).mean(),
        }
        for col in SCORES:
            for within in (False, True):
                est, lo, hi, npairs = boot(d, col, within)
                k = f"{col}_{'within' if within else 'pooled'}"
                r[k], r[k + "_lo"], r[k + "_hi"], r[k + "_pairs"] = est, lo, hi, npairs
        dl = d.dropna(subset=["lens"])
        flag = dl.lens >= FLAG
        if dl.vea.nunique() == 2 and flag.nunique() == 2:
            t = pd.crosstab(flag, dl.vea).reindex(index=[True, False], columns=[True, False], fill_value=0).to_numpy()
            r["flag_rate"] = flag.mean()
            r["or"], r["or_p"] = fisher_exact(t)
            r["vea_if_flag"], r["vea_if_not"] = dl.vea[flag].mean(), dl.vea[~flag].mean()
            tabs = []
            for _, g in dl.groupby("cond"):
                f2 = g.lens >= FLAG
                if g.vea.nunique() == 2 and f2.nunique() == 2:
                    tabs.append(pd.crosstab(f2, g.vea).reindex(index=[True, False], columns=[True, False], fill_value=0).to_numpy() + 0.5)
            if tabs:
                stt = StratifiedTable(tabs)
                r["or_mh"], r["or_mh_p"], r["or_mh_strata"] = stt.oddsratio_pooled, stt.test_null_odds().pvalue, len(tabs)
        c = d.groupby("cond").agg(vea=("vea", "mean"), lens=("lens", "mean"), prompt=("prompt", "mean"), kw=("kw_strict", "mean"))
        if len(c) >= 4:
            for col in ("lens", "prompt", "kw"):
                cc = c.dropna(subset=[col])
                if cc[col].nunique() > 1 and cc.vea.nunique() > 1:
                    rho, p = spearmanr(cc[col], cc.vea)
                    r[f"cond_rho_{col}"], r[f"cond_rho_{col}_p"] = rho, p
            r["cond_n"] = len(c)
        out.append(r)
    return out


def leak_check(run: Path, e: pd.DataFrame) -> list[dict]:
    """Identical-prompt conditions: score each episode on its OWN prompt-side reads only (keyword, strict + broad). The
    activations are a function of the prompt, so any within-condition AUROC here is engine noise or leakage through which
    positions got read (fill quotas depend on rollout length). This is why every other score is pooled per prompt."""
    from keywords import hits

    d = e[e.setting.isin(["am_main", "hp_am_fixed", "hp_abst"]) & e.vea.notna()]
    reads = pd.read_parquet(
        run / "reads.parquet", columns=["ph", "tid", "pos", "text", "ctx_left40", "ctx_right40"], filters=[("tid", "in", list(d.tid))]
    )
    h = hits(reads)
    h["tid"] = reads.tid.to_numpy()
    own = h.groupby(["tid", "pos"])[["strict_hit", "broad_hit"]].any().groupby("tid").mean()
    d = d.assign(own_strict=d.tid.map(own.strict_hit), own_broad=d.tid.map(own.broad_hit), n_own=d.n_prompt_pos_own)
    out = []
    for st, x in d.groupby("setting"):
        x = x.assign(vea=x.vea.astype(bool))
        r = {"setting": st, "n": len(x)}
        for col in ("own_strict", "own_broad", "n_own"):
            r[col], r[col + "_lo"], r[col + "_hi"], _ = boot(x.assign(ph=x.tid), col, within=True)
        out.append(r)
    return out


def contrasts(e: pd.DataFrame) -> list[dict]:
    """Test-vs-real arms: mean scores and VEA per arm, for the settings where arms are a deliberate realism contrast."""
    out = []
    ib = e[e.setting == "ib_realistic"]
    if len(ib):
        g = ib.groupby("arm").agg(
            n=("tid", "size"), labelled=("vea", lambda x: x.notna().sum()), vea=("vea", lambda x: x.astype(float).mean()),
            lens=("lens", "mean"), prompt=("prompt", "mean"), kw=("kw_strict", "mean"),
        )  # fmt: skip
        out.append({"setting": "ib_realistic", "by": "arm", "rows": g.reset_index().to_dict("records")})
    am = e[e.setting == "inspect_am"]
    if len(am):
        am = am.assign(tea=am.cond.str.endswith("|True"), prod=am.cond.str.split("|").str[3])
        g = am.groupby(["tea", "prod"]).agg(
            n=("tid", "size"), labelled=("vea", lambda x: x.notna().sum()), vea=("vea", lambda x: x.astype(float).mean()),
            lens=("lens", "mean"), prompt=("prompt", "mean"), kw=("kw_strict", "mean"),
        )  # fmt: skip
        out.append({"setting": "inspect_am", "by": "test_eval_awareness, prod", "rows": g.reset_index().to_dict("records")})
    return out


def examples(run: Path, e: pd.DataFrame, k: int = 5) -> dict:
    """Hits (lens flags, VEA), lens-only (lens flags, no VEA), misses (lens < 3, VEA): per setting, one episode per prompt."""
    lens = load_judge(run, "lens").set_index("key")
    reads = pd.read_parquet(run / "reads.parquet", columns=["ph", "tid", "pos"]).drop_duplicates(["ph", "pos"])
    src = {(r.ph, r.pos): r.tid for r in reads.itertuples()}
    first_pos = reads.groupby("ph").pos.min()
    out = {}
    for st, d in e[e.vea.notna() & e.lens.notna()].groupby("setting"):
        d = d.sample(frac=1, random_state=0)
        cats = {
            "hit": d[(d.lens >= FLAG) & d.vea.astype(bool)],
            "lens_only": d[(d.lens >= FLAG) & ~d.vea.astype(bool)],
            "miss": d[(d.lens < 3) & d.vea.astype(bool)],
        }
        out[st] = {}
        for cat, x in cats.items():
            rows = []
            for r in x.drop_duplicates("ph").head(k).itertuples():
                j = lens.loc[r.ph]
                p = evidence_pos(j.messages, j.evidence) or int(first_pos.get(r.ph, 0))
                tid = src.get((r.ph, p), r.tid)
                rows.append({
                    "tid": r.tid, "cond": r.cond, "lens": r.lens, "prompt": r.prompt, "kw_strict": r.kw_strict, "vea": bool(r.vea),
                    "evidence": j.evidence, "viewer_read": VIEW.format(tid=quote(tid), pos=p, q=quote(Q)),
                    "viewer_episode": VIEW.format(tid=quote(r.tid), pos=0, q=quote(Q)), "docent": r.docent,
                })  # fmt: skip
            out[st][cat] = rows
    return out


def fmt(x: float, lo: float | None = None, hi: float | None = None) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{x:.2f}" + (f" [{lo:.2f}, {hi:.2f}]" if lo is not None and not np.isnan(lo) else "")


def markdown(rows: list[dict], con: list[dict]) -> str:
    head = (
        "| setting | n eps | prompts | conds (multi-prompt) | VEA rate | lens AUROC within | prompt-only AUROC within | kw AUROC within "
        "| lens AUROC pooled | prompt-only AUROC pooled | kw AUROC pooled | lens≥5: VEA vs not | OR (Fisher p) | MH OR within (p) |"
    )
    L = [head, "|" + "---|" * 14]
    for r in rows:
        L.append(
            f"| {r['setting']} | {r['n']} | {r['prompts']} | {r['conds']} ({r['conds_multi_prompt']}) | {r['vea_rate']:.2f} | "
            + " | ".join(fmt(r[f"{c}_within"], r[f"{c}_within_lo"], r[f"{c}_within_hi"]) for c in ("lens", "prompt", "kw_strict"))
            + " | "
            + " | ".join(fmt(r[f"{c}_pooled"], r[f"{c}_pooled_lo"], r[f"{c}_pooled_hi"]) for c in ("lens", "prompt", "kw_strict"))
            + f" | {fmt(r.get('vea_if_flag'))} vs {fmt(r.get('vea_if_not'))} | "
            + (f"{r['or']:.2f} ({r['or_p']:.2g})" if "or" in r else "–")
            + " | "
            + (f"{r['or_mh']:.2f} ({r['or_mh_p']:.2g}, {r['or_mh_strata']} conds)" if "or_mh" in r else "–")
            + " |"
        )
    L += ["", "Across conditions (Spearman of per-condition mean score vs VEA rate):", ""]
    L += ["| setting | conds | lens ρ (p) | prompt-only ρ (p) | kw ρ (p) |", "|---|---|---|---|---|"]
    for r in rows:
        if "cond_n" in r:
            L.append(
                f"| {r['setting']} | {r['cond_n']} | "
                + " | ".join(
                    f"{r[f'cond_rho_{c}']:.2f} ({r[f'cond_rho_{c}_p']:.2g})" if f"cond_rho_{c}" in r else "–"
                    for c in ("lens", "prompt", "kw")
                )
                + " |"
            )
    for c in con:
        L += ["", f"{c['setting']} by {c['by']}:", "", "| " + " | ".join(c["rows"][0]) + " |", "|" + "---|" * len(c["rows"][0])]
        for row in c["rows"]:
            L.append("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in row.values()) + " |")
    return "\n".join(L) + "\n"


def main() -> None:
    run = Path(sys.argv[1])
    e = build(run)
    e.drop(columns=["aware"]).to_parquet(run / "scored_episodes.parquet")
    rows = setting_rows(e)
    con = contrasts(e)
    ex = examples(run, e)
    leak = leak_check(run, e)
    (run / "results.json").write_text(json.dumps({"settings": rows, "contrasts": con, "leak_check": leak}, indent=1, default=float))
    (run / "examples.json").write_text(json.dumps(ex, indent=1, default=str))
    md = markdown(rows, con)
    md += "\nLeak check (identical prompts, own reads only; within-condition AUROC, should be ~0.5):\n\n"
    md += (
        "\n".join(
            f"- {r['setting']} (n={r['n']}): "
            + ", ".join(f"{c} {fmt(r[c], r[c + '_lo'], r[c + '_hi'])}" for c in ("own_strict", "own_broad", "n_own"))
            for r in leak
        )
        + "\n"
    )
    (run / "results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
