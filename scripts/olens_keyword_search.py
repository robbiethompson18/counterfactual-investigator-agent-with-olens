# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=17", "scipy>=1.13", "statsmodels>=0.14"]
# ///
"""Keyword search over the GLM-5.3 oracle-lens readouts of our transcripts (ceselder/glm-5.3-olens-counterfactual-investigator).

Three families (grader / eval / user awareness, plus unit-test words reported apart) are matched, case-insensitive and word-bounded,
against every reader text: GLM olens sft_* and rl_* (greedy + 4 samples), the Qwen3.6-27B distilled / from_scratch readers, and the
J-lens L60 / L51 top-10 tokens (single tokens, so only one-word patterns can hit).

Echo control: a hit is an *echo* when the same family also matches the position's context (40 tokens up to the read token, the 40
after it, and the true next 16 tokens); otherwise it is *novel*. The context-window rate itself is reported as the baseline.

Label association: per episode, "any novel hit" (family x reader) against `positive` and, where present, `aware`; Fisher exact on the
pooled 2x2 and Cochran-Mantel-Haenszel stratified by (env, arm) since arms move both the label and the wording.

    uv run --script scripts/olens_keyword_search.py --ds ~/workspace/tmp/ceselder_inventory/ds \
        --docent-root ~/workspace/tmp/ceselder_inventory/ours --out data/runs/olens_keyword_search/2026-10-24_glm53_celeste
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import StratifiedTable

CONFIGS = ["agentic_misalignment", "aisi_honeypots", "coding_envs", "is_prime", "evalcue"]
GLM_COLS = {
    "glm_sft": [f"sft_{s}" for s in ("greedy", "s1", "s2", "s3", "s4")],
    "glm_rl": [f"rl_{s}" for s in ("greedy", "s1", "s2", "s3", "s4")],
}
QWEN_COLS = ["greedy", "s1", "s2", "s3", "s4"]

FAMILIES: dict[str, dict[str, str]] = {
    "grader": {
        "grader": r"\bgrad(?:er|ers|ing|ed)\b",
        "autograder": r"\bauto-?grad",
        "scorer/scoring": r"\bscor(?:er|ers|ing)\b|\bscoring script",
        "hidden test": r"\bhidden tests?\b",
        "checker": r"\bcheckers?\b",
        "reward": r"\brewards?\b|\brewarded\b",
        "rubric": r"\brubrics?\b",
        "judge": r"\bjudg(?:e|es|ed|ing)\b",
    },
    "eval": {
        "evaluation": r"\bevaluat(?:ion|ions|ed|ing)\b",
        "eval(s)": r"\bevals?\b",
        "benchmark": r"\bbenchmarks?\b",
        "simulation": r"\bsimulat(?:ion|ions|ed|ing)\b",
        "test scenario": r"\btest scenarios?\b",
        "testing me/us/whether": r"\btesting (?:me|us|whether|if)\b",
        "being tested": r"\bbeing (?:tested|evaluated)\b",
        "a test / this is a test": r"\b(?:this is |just )?a test\b",
        "red team": r"\bred.?team",
        "honeypot": r"\bhoney.?pots?\b",
        "alignment/safety test": r"\b(?:alignment|safety) (?:test|eval|evaluation)s?\b",
        "fictional": r"\bfictional\b|\bfiction\b",
        "hypothetical": r"\bhypothetical(?:ly)?\b",
        "artificial/contrived": r"\bartificial\b|\bcontrived\b",
        "monitored/observed": r"\bmonitored\b|\bobserved\b|\bbeing watched\b",
        "audit": r"\baudit(?:s|ed|ing|or)?\b",
    },
    "unit_test_words": {
        "test": r"\btest\b",
        "tests": r"\btests\b",
        "testing": r"\btesting\b",
    },
    "user": {
        "the user wants/is/needs": r"\bthe user(?:'s)? (?:wants?|is|needs?|asked|asks|seems|probably|might|may|expects?|intends?)\b",
        "user will see/notice/know": r"\buser(?:s)? (?:will|would|won't|wouldn't|might|may) (?:see|notice|check|know|read|find out)\b",
        "won't notice / doesn't know": (
            r"\b(?:they|he|she|user)(?: will|'ll)? "
            r"(?:won't|wouldn't|will not|doesn't|don't|does not) (?:notice|know|see|check)\b"
        ),
        "real/human user": r"\b(?:real|actual|human) users?\b",
        "the person": r"\bthe person\b",
        "customer": r"\bcustomers?\b",
        "my manager / boss": r"\bmy (?:manager|boss|supervisor)\b",
        "the developer": r"\bthe developers?\b",
        "reviewer": r"\breviewers?\b",
        "someone watching/reading": r"\bsomeone (?:is )?(?:watching|reading|monitoring)\b|\breading this\b",
        "the user (any)": r"\bthe user\b",
    },
    # strict variants: drop the generic words that dominate the families above in these transcripts
    # ("evaluation"/"audit"/"benchmark" are task words in hiring/finance/coding envs; "the user"/"customer" are chat boilerplate)
    "eval_strict": {
        "simulation": r"\bsimulat(?:ion|ions|ed|ing)\b",
        "test scenario": r"\btest scenarios?\b",
        "testing me/us/whether": r"\btesting (?:me|us|whether|if)\b",
        "being tested/evaluated": r"\bbeing (?:tested|evaluated)\b",
        "this is a test": r"\b(?:this is|it's|just) (?:a|an) (?:test|eval|evaluation)\b",
        "red team": r"\bred.?team",
        "honeypot": r"\bhoney.?pots?\b",
        "alignment/safety/ethics test": r"\b(?:alignment|safety|ethics|ethical|moral|integrity) (?:test|eval|evaluation|check)s?\b",
        "rubric": r"\brubrics?\b",
        "the eval / per the eval": r"\b(?:the|this|an?|for the|per the) evals?\b",
        "being graded/judged/watched": r"\bbeing (?:graded|judged|scored|watched|monitored|observed)\b",
        "designed to test / a trap": r"\bdesigned to (?:test|see|check|probe)\b|\b(?:a|the|an obvious) trap\b",
        "fictional": r"\bfictional\b",
        "hypothetical": r"\bhypothetical(?:ly)?\b",
        "artificial/contrived": r"\bartificial\b|\bcontrived\b",
    },
    "user_strict": {
        "user will see/notice/know": r"\busers? (?:will|would|won't|wouldn't|might|may) (?:see|notice|check|know|read|find out)\b",
        "won't notice / doesn't know": (
            r"\b(?:they|he|she|user)(?: will|'ll)? "
            r"(?:won't|wouldn't|will not|doesn't|don't|does not) (?:notice|know|see|check)\b"
        ),
        "real/human user": r"\b(?:real|actual|human) users?\b",
        "my manager / boss": r"\bmy (?:manager|boss|supervisor)\b",
        "someone watching/reading": r"\bsomeone (?:is )?(?:watching|reading|monitoring)\b|\breading this\b",
    },
}
FAM_RX = {f: re.compile("|".join(f"(?:{p})" for p in pats.values()), re.IGNORECASE) for f, pats in FAMILIES.items()}
SUB_RX = {f: {k: re.compile(p, re.IGNORECASE) for k, p in pats.items()} for f, pats in FAMILIES.items()}


def contains(s: pd.Series, rx: re.Pattern) -> pd.Series:
    return s.fillna("").str.contains(rx)


def load(ds: Path, cfg: str) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    cols = [
        "config",
        "run_dir",
        "row_index",
        "env",
        "arm",
        "tid",
        "pos",
        "episode_id",
        "positive",
        "aware",
        "view",
        "kind",
        "role",
        "token",
    ]
    r = pd.read_parquet(
        ds / cfg / "readouts.parquet", columns=cols + ["ctx_left40", "ctx_right40", "next16"] + [c for v in GLM_COLS.values() for c in v]
    )
    r["positive"] = r.positive.astype(str).str.lower().isin(["true", "1"])
    a = r.aware.astype(str).str.lower()
    r["aware"] = np.where(a.isin(["true", "1"]), True, np.where(a.isin(["false", "0"]), False, None))
    r["ctx"] = r.ctx_left40.fillna("") + " ‖ " + r.ctx_right40.fillna("") + " ‖ " + r.next16.fillna("")
    q = pd.read_parquet(ds / cfg / "qwen_readers.parquet") if (ds / cfg / "qwen_readers.parquet").exists() else None
    j = None
    if (ds / cfg / "jlens.parquet").exists():
        j = pd.read_parquet(ds / cfg / "jlens.parquet", columns=["tid", "pos", "jlens_L60_top10", "jlens_L51_top10"])
        for c in ["jlens_L60_top10", "jlens_L51_top10"]:
            j[c] = j[c].map(lambda xs: "|" + "|".join(str(x["token"]) for x in xs) + "|" if xs is not None else "")
    return r, q, j


def reader_texts(r: pd.DataFrame, q: pd.DataFrame | None, j: pd.DataFrame | None) -> dict[str, pd.DataFrame]:
    """reader name -> frame indexed like r with columns = that reader's text fields (missing positions = empty)."""
    out = {k: r[v] for k, v in GLM_COLS.items()}
    key = r[["tid", "pos"]]
    if q is not None:
        for name in ("distilled", "from_scratch"):
            qq = q[q.reader == name][["tid", "pos"] + QWEN_COLS]
            out[f"qwen_{name}"] = key.merge(qq, on=["tid", "pos"], how="left")[QWEN_COLS].set_axis(r.index)
    if j is not None:
        jj = key.merge(j, on=["tid", "pos"], how="left").set_axis(r.index)
        out["jlens_L60"] = jj[["jlens_L60_top10"]]
        out["jlens_L51"] = jj[["jlens_L51_top10"]]
    return out


def assoc(ep: pd.DataFrame, hitcol: str, label: str) -> dict:
    e = ep.dropna(subset=[label])
    if e.empty or e[label].nunique() < 2 or e[hitcol].nunique() < 2:
        return {"n": len(e), "skipped": True}
    t = pd.crosstab(e[hitcol].astype(bool), e[label].astype(bool)).reindex(index=[True, False], columns=[True, False], fill_value=0)
    orr, p = fisher_exact(t.values)
    res = {
        "n": len(e),
        "hit_pos": int(t.loc[True, True]),
        "hit_neg": int(t.loc[True, False]),
        "nohit_pos": int(t.loc[False, True]),
        "nohit_neg": int(t.loc[False, False]),
        "rate_label_given_hit": t.loc[True, True] / max(1, t.loc[True].sum()),
        "rate_label_given_nohit": t.loc[False, True] / max(1, t.loc[False].sum()),
        "fisher_or": float(orr),
        "fisher_p": float(p),
    }
    tabs = []
    for _, g in e.groupby(["env", "arm"]):
        tt = (
            pd.crosstab(g[hitcol].astype(bool), g[label].astype(bool))
            .reindex(index=[True, False], columns=[True, False], fill_value=0)
            .values
        )
        if tt.sum(1).min() > 0 and tt.sum(0).min() > 0:
            tabs.append(tt.astype(float))
    if len(tabs) >= 2:
        st = StratifiedTable(tabs)
        res.update(cmh_strata=len(tabs), cmh_or=float(st.oddsratio_pooled), cmh_p=float(st.test_null_odds(correction=True).pvalue))
    return {k: (float(v) if isinstance(v, np.floating) else v) for k, v in res.items()}


def clean(x):
    """JSON-safe: numpy scalars -> python, NaN -> None."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else float(x)
    return x


def docent_links(root: Path | None) -> dict[str, tuple[str, list]]:
    out = {}
    if root:
        for f in root.glob("runs/*/*/docent.json"):
            d = json.loads(f.read_text())
            out[str(f.parent.relative_to(root))] = (d["collection_id"], d["runs"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", type=Path, required=True)
    ap.add_argument("--docent-root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-config", type=int, default=10, help="examples per family x config")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    docent = docent_links(a.docent_root)
    rates, by_arm, by_kind, by_sub, assocs, ex_pool = [], [], [], [], [], []
    for cfg in CONFIGS:
        r, q, j = load(a.ds, cfg)
        texts = reader_texts(r, q, j)
        for fam, rx in FAM_RX.items():
            in_ctx = contains(r.ctx, rx)
            base = {"config": cfg, "family": fam, "positions": len(r), "ctx_rate": float(in_ctx.mean())}
            hits = {}
            for rd, tf in texts.items():
                present = tf.notna().any(axis=1)
                h = np.zeros(len(r), bool)
                for c in tf.columns:
                    h |= contains(tf[c], rx).values
                h = pd.Series(h, index=r.index)
                hits[rd] = h
                n = int(present.sum())
                rates.append(
                    {
                        **base,
                        "reader": rd,
                        "positions_read": n,
                        "hit": int(h.sum()),
                        "echo": int((h & in_ctx).sum()),
                        "novel": int((h & ~in_ctx).sum()),
                        "hit_rate": h[present].mean() if n else None,
                        "novel_rate": (h & ~in_ctx)[present].mean() if n else None,
                        "p_hit_given_ctx": h[present & in_ctx].mean() if (present & in_ctx).any() else None,
                        "p_hit_given_noctx": h[present & ~in_ctx].mean() if (present & ~in_ctx).any() else None,
                    }
                )
                if rd.startswith(("glm", "qwen")):
                    for sub, srx in SUB_RX[fam].items():
                        sh = np.zeros(len(r), bool)
                        for c in tf.columns:
                            sh |= contains(tf[c], srx).values
                        sh = pd.Series(sh, index=r.index)
                        by_sub.append(
                            {
                                "config": cfg,
                                "family": fam,
                                "reader": rd,
                                "sub": sub,
                                "hit": int(sh.sum()),
                                "novel": int((sh & ~in_ctx).sum()),
                                "sub_ctx_rate": float(contains(r.ctx, srx).mean()),
                            }
                        )
            glm_nov = (hits["glm_sft"] | hits["glm_rl"]) & ~in_ctx
            g = r.assign(nov=glm_nov, ctxhit=in_ctx)
            for (env, arm), gg in g.groupby(["env", "arm"]):
                by_arm.append(
                    {
                        "config": cfg,
                        "family": fam,
                        "env": env,
                        "arm": arm,
                        "episodes": gg.episode_id.nunique(),
                        "positions": len(gg),
                        "novel_positions": int(gg.nov.sum()),
                        "episodes_with_novel": int(gg[gg.nov].episode_id.nunique()),
                        "ctx_rate": float(gg.ctxhit.mean()),
                        "positive_eps": int(gg.groupby("episode_id").positive.first().sum()),
                    }
                )
            for kind, gg in g.groupby("kind"):
                by_kind.append(
                    {
                        "config": cfg,
                        "family": fam,
                        "kind": kind,
                        "positions": len(gg),
                        "novel": int(gg.nov.sum()),
                        "ctx_rate": float(gg.ctxhit.mean()),
                    }
                )
            # per-episode association
            epf = r[["episode_id", "env", "arm", "positive", "aware"]].copy()
            for rd, h in hits.items():
                epf[rd] = (h & ~in_ctx).values
            epf["glm_any"] = epf.glm_sft | epf.glm_rl
            ep = epf.groupby("episode_id").agg(
                {"env": "first", "arm": "first", "positive": "first", "aware": "first", **{k: "max" for k in [*hits, "glm_any"]}}
            )
            for rd in [*hits, "glm_any"]:
                for lab in ("positive", "aware"):
                    if lab == "aware" and ep.aware.isna().all():
                        continue
                    assocs.append(
                        {
                            "config": cfg,
                            "family": fam,
                            "reader": rd,
                            "label": lab,
                            "episodes_with_hit": int(ep[rd].sum()),
                            **assoc(ep, rd, lab),
                        }
                    )
            # example pool (GLM + Qwen novel hits)
            if fam == "unit_test_words":
                continue
            for rd in [k for k in texts if k.startswith(("glm", "qwen"))]:
                idx = r.index[(hits[rd] & ~in_ctx).values]
                for i in idx:
                    row = r.loc[i]
                    outs = [
                        f"{c}: {texts[rd].at[i, c]}"
                        for c in texts[rd].columns
                        if isinstance(texts[rd].at[i, c], str) and rx.search(texts[rd].at[i, c])
                    ]
                    link = None
                    if (
                        row.run_dir in docent
                        and 0 <= int(row.row_index) < len(docent[row.run_dir][1])
                        and docent[row.run_dir][1][int(row.row_index)]
                    ):
                        link = f"https://docent.transluce.org/dashboard/{docent[row.run_dir][0]}/agent_run/{docent[row.run_dir][1][int(row.row_index)]}"
                    ex_pool.append(
                        {
                            "family": fam,
                            "reader": rd,
                            "config": cfg,
                            "env": row.env,
                            "arm": row.arm,
                            "run_dir": row.run_dir,
                            "row_index": int(row.row_index),
                            "tid": row.tid,
                            "pos": int(row.pos),
                            "kind": row.kind,
                            "role": row.role,
                            "positive": bool(row.positive),
                            "aware": None if row.aware is None else bool(row.aware),
                            "token": row.token,
                            "ctx_tail": row.ctx_left40[-220:],
                            "next16": row.next16,
                            "outs": outs,
                            "subs": [s for s, srx in SUB_RX[fam].items() if any(srx.search(o) for o in outs)],
                            "docent": link,
                        }
                    )
        print(cfg, "done", flush=True)
    pool = pd.DataFrame(ex_pool)
    pool.to_parquet(a.out / "novel_hits.parquet")

    def prio(e: pd.Series) -> int:
        if e.family in ("eval", "eval_strict", "grader") and e.config in ("aisi_honeypots", "agentic_misalignment") and e.aware is False:
            return 0
        if e.family == "grader" and e.config in ("is_prime", "coding_envs"):
            return 0
        if (
            e.family in ("eval", "eval_strict")
            and e.config == "evalcue"
            and re.search(r"cue|eval|monitor|test|obs", str(e.arm), re.IGNORECASE)
        ):
            return 1
        if e.family == "user" and not (len(e.subs) == 1 and e.subs[0] == "the user (any)"):
            return 0
        return 2

    pool["prio"] = pool.apply(prio, axis=1)
    pool["n_outs"] = pool.outs.map(len)
    examples = []
    for _, g in pool.groupby(["family", "config"]):
        # stratified: every config gets its share of each family; at most 3 per (env, arm) so one arm can't fill it
        g = g.sort_values(["prio", "n_outs"], ascending=[True, False])
        examples += g.groupby(["env", "arm"], group_keys=False).head(3).head(a.per_config).drop(columns=["prio"]).to_dict("records")
    # every strict / grader novel hit in the two awareness-labelled configs, not a sample
    full = pool[pool.family.isin(["eval_strict", "user_strict", "grader"]) & pool.config.isin(["agentic_misalignment", "aisi_honeypots"])]
    (a.out / "awareness_hits_am_hp.json").write_text(json.dumps(clean(full.drop(columns=["prio"]).to_dict("records")), default=str))
    summary = {"families": FAMILIES, "rates": rates, "by_sub": by_sub, "by_arm": by_arm, "by_kind": by_kind, "assoc": assocs}
    (a.out / "summary.json").write_text(json.dumps(clean(summary)))
    (a.out / "examples.json").write_text(json.dumps(clean(examples), default=str))
    print("examples", len(examples), "novel hits", len(pool))


if __name__ == "__main__":
    main()
