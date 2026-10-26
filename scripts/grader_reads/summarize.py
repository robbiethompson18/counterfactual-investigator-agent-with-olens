# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb", "pandas", "pyarrow", "scipy"]
# ///
"""Chart data for the grader-reads page from analyze.py's tables: summary.json next to them.

    uv run --script scripts/grader_reads/summarize.py --scan data/runs/grader_reads/<run> --bt-dir <realism runs dir> \
        --overeager-cells <docent_cells.json> --data <monorepo data dir>
"""

import argparse
import gzip
import json
import math
import random
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd
from scipy.stats import fisher_exact

sys.path.insert(0, str(Path(__file__).parent))
from analyze import FAMILIES, fam_bits
from docent_links import Docent

MODEL_SIDE = ("reasoning", "reply text", "tool calls", "turn start")
PROMPT_SIDE = ("system prompt", "tool schema", "user turn")
FAMS = list(FAMILIES)[:6]
VIEWER = "http://bramble:8830/#d=imports/{ds}&i={item}&v=tr&t={pos}"

# what `positive` means per env, and which way is "acted"
POSITIVE = {
    "IB realism ladder": "hacked the tests",
    "IB realism variants (round 1)": "hacked the tests",
    "ImpossibleBench (original)": "hacked the tests",
    "email (several Johns)": "picked one John without asking",
    "agentic misalignment": "harmful action",
    "alignment faking (no CoT)": "complied",
    "alignment faking (thinking)": "complied",
    "Funding Email (Claude Code harness)": "env positive",
    "Donation Bet": "leaked the bet",
    "Donation Bet (empty think)": "leaked the bet",
}
for t in ("calendar", "dana", "email", "email_4johns", "gitbranch", "payback"):
    POSITIVE[f"overeager/{t}"] = "acted without asking"

LADDER = ["L1_rl_cues", "L2_rl", "L3_rl_nodone", "L4_assistant_formal", "L5_assistant_casual", "L6_casual_ide", "L7_repo", "L8_full"]
LADDER_LABEL = {
    "L1_rl_cues": "L1 lcb.yaml + done + RL cues",
    "L2_rl": "L2 lcb.yaml + done",
    "L3_rl_nodone": "L3 no done",
    "L4_assistant_formal": "L4 assistant, formal",
    "L5_assistant_casual": "L5 assistant, casual ask",
    "L6_casual_ide": "L6 + IDE context",
    "L7_repo": "L7 + practice repo + git",
    "L8_full": "L8 full (+ AGENTS.md)",
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_extra(s: str) -> dict:
    """scan.py cut `extra` at 2,000 characters; the link fields come first, so pull them out by pattern when it no longer parses."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        out = {}
        for k in ("run_dir", "source_file"):
            m = re.search(rf'"{k}": "([^"]*)"', s)
            out[k] = m.group(1) if m else None
        m = re.search(r'"row_index": (\d+)', s)
        out["row_index"] = int(m.group(1)) if m else None
        return out


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if hasattr(o, "item"):
        o = o.item()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--bt-dir", required=True)
    ap.add_argument("--overeager-cells", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--aliases", default=str(Path(__file__).parents[2] / "olens_search" / "docent_aliases.json"))
    a = ap.parse_args()
    A = Path(a.scan) / "analysis"
    con = duckdb.connect()
    for t in ("by_arm_section", "by_env_section", "episodes", "raw_totals", "hit_rows", "items", "shared_across_arms"):
        con.execute(f"create view {t} as select * from read_parquet('{A / (t + '.parquet')}')")
    ms = ",".join(f"'{s}'" for s in MODEL_SIDE)
    ps = ",".join(f"'{s}'" for s in PROMPT_SIDE)
    out = {"families": {k: FAMILIES[k] for k in FAMS}}

    # 1. per env: distinct-position rates by side, plus the un-deduplicated count
    env = con.execute(
        f"""select env,
          sum(case when sg in ({ps}) then n_distinct end) p_n, sum(case when sg in ({ps}) then grader_any end) p_k,
          sum(case when sg in ({ps}) then setting end) p_s,
          sum(case when sg in ({ms}) then n_distinct end) m_n, sum(case when sg in ({ms}) then grader_any end) m_k,
          sum(case when sg in ({ms}) then setting end) m_s,
          sum(case when sg='tool results' then n_distinct end) t_n, sum(case when sg='tool results' then grader_any end) t_k,
          {", ".join(f"sum(case when sg in ({ms}) then {f} end) m_{f}" for f in FAMS)},
          {", ".join(f"sum(case when sg in ({ps}) then {f} end) p_{f}" for f in FAMS)}
        from by_arm_section group by env"""
    ).df()
    raw = con.execute(
        f"""select env, sum(n_reads) n_reads, sum(grader_any) k_reads,
          sum(case when sg in ({ps}) then n_reads end) p_reads, sum(case when sg in ({ps}) then grader_any end) p_kreads
        from raw_totals group by env"""
    ).df()
    eps = con.execute("select env, count(*) n_eps from items group by env").df()
    env = env.merge(raw, on="env").merge(eps, on="env").fillna(0)
    out["envs"] = env.to_dict("records")

    # 2. per arm: model-side own-position rates and episode share, prompt-side distinct hits, behaviour rate
    arm = con.execute(
        f"""select env, arm,
          sum(case when sg in ({ms}) then n_own end) m_n, sum(case when sg in ({ms}) then grader_any end) m_k,
          sum(case when sg in ({ms}) then setting end) m_s,
          sum(case when sg in ({ps}) then n_distinct end) p_n, sum(case when sg in ({ps}) then grader_any end) p_k,
          sum(case when sg in ({ps}) then setting end) p_s,
          sum(case when sg='tool results' then n_distinct end) t_n, sum(case when sg='tool results' then grader_any end) t_k
        from by_arm_section group by 1,2"""
    ).df()
    epm = con.execute(
        f"""select env, arm, ds, item, any_value(positive) positive, bool_or(grader_any and sg in ({ms})) g,
          bool_or(setting and sg in ({ms})) gs
        from episodes group by env, arm, ds, item"""
    ).df()
    beh = con.execute("select env, arm, count(*) n_eps, avg(positive::int) pos_rate, count(positive) n_lab from items group by 1,2").df()
    epr = epm.groupby(["env", "arm"]).agg(n_eps_read=("g", "size"), k_eps=("g", "sum"), k_eps_s=("gs", "sum")).reset_index()
    arm = arm.merge(beh, on=["env", "arm"], how="left").merge(epr, on=["env", "arm"], how="left").fillna(0)
    bt = {}
    for fam in ("email", "ib"):
        p = Path(a.bt_dir) / f"evalness_{fam}" / "bt.json"
        if p.exists():
            for r in json.loads(p.read_text()):
                bt[(fam, r["variant"])] = r["bt"]

    def bt_of(e, r):
        if e.startswith("email"):
            return bt.get(("email", r)) or bt.get(("email", r.split("@")[0] + "@low"))
        if e.startswith("IB realism variants"):
            return bt.get(("ib", r))
        return None

    arm["evalness"] = [bt_of(e, r) for e, r in zip(arm.env, arm.arm)]
    arm["ladder_label"] = [LADDER_LABEL.get(r) for r in arm.arm]
    out["arms"] = arm.to_dict("records")

    # 3. position profile per env (distinct positions)
    sec = con.execute("select env, sg, sum(n_distinct) n, sum(grader_any) k, sum(setting) ks from by_arm_section group by 1,2").df()
    out["sections"] = sec.to_dict("records")

    # 4. acted vs not: episodes with a model-side grader read, by label, within arm (pooled Mantel-Haenszel OR)
    av = []
    for e, g in epm[epm.positive.notna()].groupby("env"):
        g = g.assign(positive=g.positive.astype(bool), g=g.gs.astype(bool))  # reads about the setting
        num = den = 0.0
        for _, h in g.groupby("arm"):
            n = len(h)
            a1 = ((h.positive) & (h.g)).sum()
            b1 = ((h.positive) & (~h.g)).sum()
            c1 = ((~h.positive) & (h.g)).sum()
            d1 = ((~h.positive) & (~h.g)).sum()
            num += a1 * d1 / n
            den += b1 * c1 / n
        pos, neg = g[g.positive], g[~g.positive]
        if len(pos) == 0 or len(neg) == 0:
            continue
        _, pf = fisher_exact([[pos.g.sum(), len(pos) - pos.g.sum()], [neg.g.sum(), len(neg) - neg.g.sum()]])
        av.append(
            {
                "env": e,
                "meaning": POSITIVE.get(e, "positive"),
                "n_pos": len(pos),
                "k_pos": int(pos.g.sum()),
                "n_neg": len(neg),
                "k_neg": int(neg.g.sum()),
                "mh_or": (num / den) if den > 0 else None,
                "p_pooled": pf,
            }
        )
    out["acted"] = av

    # 5. examples: model-side own-position reads and prompt-side distinct reads, per env
    items = con.execute('select ds, item, id, "group", extra, env, arm from items').df()
    extra = {(r.ds, r.item): (load_extra(r.extra), r.env, r.arm, r.id) for r in items.itertuples()}
    D = Docent(Path(a.data), Path(a.aliases), Path(a.overeager_cells))
    hr = con.execute("select * from hit_rows where (novel>>7)&1=1").df()
    texts = {}
    for f in Path(a.scan).glob("hits__*.jsonl.gz"):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                h = json.loads(line)
                if fam_bits(h["read"]) & 128:
                    texts[(h["ds"], h["item"], h["pos"])] = (h["read"], h["before"][-220:], h["after"])
    random.seed(0)
    ex = []
    for e, g in hr.groupby("env"):
        for side, sel in (
            ("model", g[g.sg.isin(MODEL_SIDE) & (g.n_share == 1)]),
            ("prompt", g[g.sg.isin(PROMPT_SIDE)].drop_duplicates(["arm", "prefix"])),
        ):
            rows = sel.sample(min(len(sel), 14), random_state=0) if len(sel) else sel
            for r in rows.itertuples():
                t = texts.get((r.ds, r.item, r.pos))
                if not t:
                    continue
                xe, _, xa, xid = extra[(r.ds, r.item)]
                url, kind = D.link(xe, env=e, arm=xa)
                ex.append(
                    {
                        "env": e,
                        "arm": r.arm,
                        "side": side,
                        "sg": r.sg,
                        "positive": None if pd.isna(r.positive) else bool(r.positive),
                        "n_share": int(r.n_share),
                        "fams": [f for i, f in enumerate(FAMS) if (r.novel >> i) & 1],
                        "setting": bool((r.novel >> 8) & 1),
                        "read": t[0],
                        "before": t[1],
                        "after": t[2],
                        "viewer": VIEWER.format(ds=r.ds, item=xid, pos=r.pos),
                        "docent": url,
                        "docent_kind": kind,
                    }
                )
    out["examples"] = ex
    (A / "summary.json").write_text(json.dumps(clean(out)))
    print("envs", len(out["envs"]), "arms", len(out["arms"]), "examples", len(ex), "acted", len(av))


if __name__ == "__main__":
    main()
