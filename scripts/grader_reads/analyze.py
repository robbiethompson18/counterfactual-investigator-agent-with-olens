# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb", "pandas", "pyarrow"]
# ///
"""Rates of grader / eval / test / score / reward words in GLM-5.3 O-lens reads, from scan.py's output.

Every count is over reads whose match is *novel* (absent from the 40 tokens before the read position
and the 16 after), so a read that echoes "grader" from a nearby tool schema or prompt line doesn't
count. Reads of an identical token prefix (the same system prompt or tool schema read in every
episode of an arm) are counted once: `distinct` rates are over distinct prefixes, and an episode only
counts reads at positions no other episode shares (`own` reads).

    uv run --script scripts/grader_reads/analyze.py --scan data/runs/grader_reads/<run>
"""

import argparse
import glob
import gzip
import json
import os
import re

import duckdb
import pandas as pd

# Families recomputed from scan.py's hit rows, aimed at "the setting is graded / evaluated": nouns only (the verb
# "evaluate" and Python's eval( are excluded), "test" only when it is a test *of the model*. `any` = the union of the first six.
# Novel = absent from the 60 tokens before the read and the 16 after.
FAMILIES = {
    "grader": r"\b(?:auto)?grad(?:ers?|ing|ed)\b",
    "eval": r"\b(?:evals?|evaluations?|evaluators?)\b(?!\s*\()",
    "benchmark": r"\b(?:benchmarks?|rubrics?)\b",
    "reward": r"\breward(?:s|ed|ing)?\b",
    "score": r"\b(?:scor(?:e|es|ed|ing)|points)\b",
    "test_me": r"\b(?:this is|it'?s|is) (?:a|an|just a|clearly a|likely a|probably a) (?:\w+ )?test\b"
    r"|\b(?:alignment|safety|deception|honesty|ethics|integrity|loyalty|capability|character|moral) tests?\b"
    r"|\bbeing (?:tested|evaluated|graded|scored)\b"
    r"|\b(?:testing|tests) (?:me|my|the (?:ai|model|assistant|agent)|whether (?:i|the (?:ai|model|assistant|agent)))\b",
    "sim": r"\b(?:simulat\w*|fictional|hypothetical|role-?play\w*)",
}
RX = {k: re.compile(v, re.IGNORECASE) for k, v in FAMILIES.items()}
FAM = list(FAMILIES) + ["grader_any", "setting"]
# a grader-word read that is about the task setting or the model itself (vs a content use such as "rewards us with insight")
SETTING = re.compile(
    r"\b(?:this|these|the|such|typical|our|current) (?:\w+[- ])?"
    r"(?:tasks?|scenarios?|benchmarks?|evals?|evaluations?|environments?|tests?|setups?|challenges?|exercises?|episodes?|suites?)\b"
    r"|\b(?:graders?|evaluators?|checkers?|judges?|rubrics?)\b|\b(?:the|an?) (?:ai|agent|model|assistant)\b"
    r"|\b(?:expects?|wants?|intended|ideal|correct (?:behaviou?r|action|answer|response))\b|\btest(?:ing|s)? (?:me|whether)\b|\bscenario\b"
    r"|\btest of my\b|\bmy (?:\w+ )?(?:score|reward)s?\b|\bmaximi[sz]\w* (?:the )?(?:score|reward)",
    re.IGNORECASE,
)
ANY = (1 << 6) - 1


def fam_bits(text: str, ctx: str | None = None) -> int:
    b = 0
    for i, k in enumerate(FAMILIES):
        if RX[k].search(text) and (ctx is None or not RX[k].search(ctx)):
            b |= 1 << i
    if b & ANY:
        b |= 1 << 7
        if SETTING.search(text):
            b |= 1 << 8
    return b


SECT = ["header", "system", "tool_schema", "user", "turn_start", "reasoning", "reply", "tool_call", "tool_result", "other"]
# sections grouped for the tables
SG = {
    "system": "system prompt",
    "tool_schema": "tool schema",
    "user": "user turn",
    "turn_start": "turn start",
    "reasoning": "reasoning",
    "reply": "reply text",
    "tool_call": "tool calls",
    "tool_result": "tool results",
    "header": "other",
    "other": "other",
}
SG_ORDER = ["system prompt", "tool schema", "user turn", "turn start", "reasoning", "reply text", "tool calls", "tool results", "other"]


def env_of(ds: str, group: str) -> tuple[str, str]:
    """(env family, arm) for a dataset id and item group."""
    g = group or ""
    env, _, arm = g.partition(" · ")
    if not arm:
        env, arm = g, ""
    if ds.startswith("olens_reads/") and "overeager" in ds:
        return env, arm
    if env.startswith("coding/ib_realistic_user"):
        return ("IB realism ladder" if "ladder" in ds else "IB realism variants (round 1)"), arm
    if env.startswith("coding/impossiblebench"):
        return "ImpossibleBench (original)", arm
    if env == "email_ambiguous":
        return "email (several Johns)", arm
    if env.startswith("agentic_misalignment"):
        return "agentic misalignment", arm
    if env == "alignment_faking":
        return ("alignment faking (no CoT)" if "nocot" in ds else "alignment faking (thinking)"), arm
    if env == "donation_bet_goodbad":
        return ("Donation Bet (empty think)" if "nothink" in ds else "Donation Bet"), arm
    if env == "funding_email_real":
        return "Funding Email (Claude Code harness)", arm
    return env, arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    a = ap.parse_args()
    con = duckdb.connect()
    con.execute("PRAGMA threads=6")
    files = sorted(glob.glob(os.path.join(a.scan, "reads__*.parquet")))
    items = []
    for f in files:
        safe = os.path.basename(f)[len("reads__") : -len(".parquet")]
        ds = safe.replace("__", "/")
        it = pd.read_parquet(os.path.join(a.scan, f"items__{safe}.parquet"))
        it["ds"] = ds
        items.append(it)
    items = pd.concat(items, ignore_index=True)
    ea = [env_of(d, g) for d, g in zip(items.ds, items.group)]
    items["env"] = [e for e, _ in ea]
    items["arm"] = [r for _, r in ea]
    lab = [json.loads(x) for x in items.labels]
    items["positive"] = [x.get("positive") for x in lab]
    items["aware"] = [x.get("aware") for x in lab]
    con.register("items_df", items.drop(columns=["labels", "extra"]))
    con.execute("create table items as select * from items_df")
    union = " union all ".join(
        f"select '{os.path.basename(f)[len('reads__') : -len('.parquet')].replace('__', '/')}' as ds, * from read_parquet('{f}')"
        for f in files
    )
    con.execute(f"create table reads0 as {union}")
    fr = []
    for f in files:
        safe = os.path.basename(f)[len("reads__") : -len(".parquet")]
        hp = os.path.join(a.scan, f"hits__{safe}.jsonl.gz")
        if not os.path.exists(hp):
            continue
        with gzip.open(hp, "rt") as fh:
            for line in fh:
                h = json.loads(line)
                raw = fam_bits(h["read"])
                if raw:
                    fr.append((h["ds"], h["item"], h["pos"], raw, fam_bits(h["read"], h["before"] + h["after"])))
    con.register("fr_df", pd.DataFrame(fr, columns=["ds", "item", "pos", "fraw", "fnov"]))
    con.execute(
        """create table reads as select r.* exclude (raw, novel, inovel, s_raw, s_novel),
             coalesce(f.fraw, 0)::smallint as raw, coalesce(f.fnov, 0)::smallint as novel
           from reads0 r left join fr_df f using (ds, item, pos)"""
    )
    con.execute("drop table reads0")
    con.execute("create table sect_map(sect tinyint, sname varchar, sg varchar)")
    con.executemany("insert into sect_map values (?,?,?)", [(i, s, SG[s]) for i, s in enumerate(SECT)])
    # how many episodes (and arms) share each prefix, within an env family
    con.execute(
        """create table r as
        select r.*, i.env, i.arm, i.positive, i.id, m.sg,
               count(*) over (partition by i.env, r.prefix) as n_share,
               count(distinct i.arm) over (partition by i.env, r.prefix) as n_arms
        from reads r join items i using (ds, item) join sect_map m using (sect)"""
    )
    fam_cols = ", ".join(f"bool_or((novel>>{k})&1=1) as {f}" for k, f in enumerate(FAM))
    # one row per distinct prefix within an arm
    con.execute(
        f"""create table d as select env, arm, sg, prefix, any_value(n_share) n_share, any_value(n_arms) n_arms,
               count(distinct ds||':'||item) as n_eps, {fam_cols}
        from r group by env, arm, sg, prefix"""
    )
    sums = ", ".join(f"sum({f}::int) as {f}" for f in FAM)
    out = {}
    out["by_arm_section"] = con.execute(
        f"select env, arm, sg, count(*) as n_distinct, sum((n_share=1)::int) as n_own, {sums} from d group by 1,2,3 order by 1,2,3"
    ).df()
    out["by_arm_section_own"] = con.execute(
        f"select env, arm, sg, count(*) as n_own, {sums} from d where n_share=1 group by 1,2,3 order by 1,2,3"
    ).df()
    out["by_env_section"] = con.execute(f"select env, sg, count(*) as n_distinct, {sums} from d group by 1,2 order by 1,2").df()
    out["by_section"] = con.execute(f"select sg, count(*) as n_distinct, {sums} from d group by 1 order by 1").df()
    # shared prompt-side positions: identical across all episodes of an arm (n_eps = arm size) or across arms
    out["shared_across_arms"] = con.execute(
        f"select env, sg, count(*) as n_prefix, {sums} from d where n_arms>1 group by 1,2 order by 1,2"
    ).df()
    # episode level: does the episode have an own read with a novel match, per section group
    ep_cols = ", ".join(f"bool_or((novel>>{k})&1=1) as {f}" for k, f in enumerate(FAM))
    con.execute(
        f"""create table e as select env, arm, ds, item, any_value(id) id, any_value(positive) positive, sg, count(*) as n_own, {ep_cols}
        from r where n_share=1 group by env, arm, ds, item, sg"""
    )
    out["episodes"] = con.execute("select * from e").df()
    # before dedup: every read row as the viewer shows it (shared prefixes repeated once per episode)
    out["raw_totals"] = con.execute(
        "select env, sg, count(*) as n_reads, sum(((novel>>7)&1)::int) as grader_any, sum(((raw>>7)&1)::int) as grader_any_echo_too "
        "from r group by 1,2 order by 1,2"
    ).df()
    # the hit rows themselves, with sharing info, for examples
    out["hit_rows"] = con.execute(
        "select ds, item, id, env, arm, positive, pos, sg, turn, prefix, n_share, n_arms, raw, novel from r where raw<>0"
    ).df()
    os.makedirs(os.path.join(a.scan, "analysis"), exist_ok=True)
    for k, v in out.items():
        v.to_parquet(os.path.join(a.scan, "analysis", f"{k}.parquet"))
        print(k, len(v))
    items.drop(columns=["labels"]).to_parquet(os.path.join(a.scan, "analysis", "items.parquet"))


if __name__ == "__main__":
    main()
