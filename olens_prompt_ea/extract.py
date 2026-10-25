# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.1", "pandas>=2.2", "pyarrow>=17"]
# ///
"""Prompt-side O-lens reads per episode, from the olens_search DuckDB (olens_search/build_db.py).

The prompt is everything before the first assistant turn: system prompt (incl. tool schemas) and user turn. For AISI
Misalignment Continuation the scripted assistant prefill turns are therefore NOT prompt. A read
position is prompt-side when its token ends at or before that point.

Identical prompts give identical activations, so reads are pooled per unique prompt (`ph` = md5 of the rendered prompt),
across every episode with that prompt. Episodes that share a prompt get the same lens score; this also removes a leak:
which prompt positions got read in an episode depends on how long its rollout was (fill quotas).

Writes into OUT:
  episodes.parquet  one row per episode: setting, cond, ph, labels, generated text (for VEA labelling), links
  prompts.parquet   one row per unique prompt: rendered prompt text, n read positions
  reads.parquet     one row per (ph, pos, reader text): GLM O-lens SFT/RL greedy + 4 samples, token, context

    uv run --script olens_prompt_ea/extract.py --db data/olens_search/olens.duckdb --out data/runs/olens_prompt_ea/<run>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import duckdb
import pandas as pd

# config -> setting name; None = skipped (see docs/olens-prompt-eval-awareness.md for why)
SETTINGS = {
    "agentic_misalignment": "am_main",
    "aisi_honeypots": None,  # split by env below
    "cfi_ib_realistic": "ib_realistic",
    "cfi_donation_bet": "donation_bet",
    "donation_bet_goodbad": "donation_bet",  # same data, renamed in the dashboard build of 2026-10-24 18:32 UTC
    "ib_realistic_user_partial": "ib_realistic",
    "evalcue": "evalcue",
    "inspect_agentic_misalignment": "inspect_am",
    "inspect_sad_oversight": "sad_oversight",
    "inspect_instrumentaleval": "instrumentaleval",
    "inspect_agentharm": "agentharm",
    "inspect_mask": "mask",
}
HONEYPOT = {"aisi_honeypots/am": "hp_am_fixed", "aisi_honeypots/mc": "hp_mc", "aisi_honeypots/abst": "hp_abst"}
# evalcue envs grouped into families; cue = envs built around an eval / grading / metadata cue
EVALCUE_FAMILY = {
    "scheming": "ec_scheming",
    "shutdown": "ec_shutdown",
    "whistleblow": "ec_whistleblow",
    "sandbagging": "ec_sandbagging",
    "overnight": "ec_overnight",
    "hiring": "ec_hiring",
    "dumb": "ec_dumb_user",
    "hidden": "ec_hidden_goal",
    **{k: "ec_cue" for k in ("mmlu", "hinted", "lcb", "secret", "summ", "precommit", "repo", "exclude", "git")},
}


def evalcue_setting(env: str) -> str:
    return EVALCUE_FAMILY.get(env.split("_")[0], "ec_other")


MARK = re.compile(r"<\|(assistant|user|system|observation)\|>")


MC_INCLUDE_PREFILL = False  # --mc-include-prefill: AISI MC prompt runs to the LAST assistant turn (scripted prefill counted)


def prompt_end(text: str, setting: str) -> int:
    """Char offset of the first assistant turn: the prompt is everything before it (Kylie's definition, 2026-10-24). For
    AISI Misalignment Continuation this excludes the scripted assistant prefill turns, which count as rollout context.
    Inspect proxy transcripts: call 0, whose first assistant marker is its completion. With MC_INCLUDE_PREFILL the MC prompt
    also takes in the scripted assistant turns (only the last assistant turn is GLM's)."""
    if MC_INCLUDE_PREFILL and setting == "hp_mc":
        return text.rfind("<|assistant|>")
    return text.find("<|assistant|>")


def generated(text: str, start: int, tool_cap: int = 300) -> str:
    """The model's own text after the prompt: assistant turns verbatim (reasoning kept), tool outputs and later user turns
    shortened. This is what the VEA monitor reads."""
    out, parts = [], MARK.split(text[start:])
    # parts = ["", role, body, role, body, ...]
    for role, body in zip(parts[1::2], parts[2::2]):
        body = body.replace("[gMASK]<sop>", "")
        if role == "assistant":
            out.append(f"[assistant]\n{body.strip()}")
        elif body.strip():
            b = body.strip()
            out.append(f"[{role}] {b[:tool_cap]}{' …[' + str(len(b) - tool_cap) + ' chars cut]' if len(b) > tool_cap else ''}")
    return "\n\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mc-include-prefill", action="store_true", help="count AISI MC's scripted assistant turns as prompt")
    a = ap.parse_args()
    global MC_INCLUDE_PREFILL
    MC_INCLUDE_PREFILL = a.mc_include_prefill
    a.out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(a.db), read_only=True)
    cfgs = ", ".join(f"'{c}'" for c in SETTINGS)
    meta = con.execute(f"""
        select tid, any_value(config) config, any_value(env) env, any_value(arm) arm, any_value(item) item,
               any_value(sample) sample, any_value(aware) aware, any_value(positive) positive, any_value(labels) labels,
               any_value(run_dir) run_dir, any_value(row_index) row_index, any_value(episode_id) episode_id
        from positions where config in ({cfgs}) and tid not like '%:late' group by tid""").df()
    eps = con.execute(f"select tid, rendered_text, truncated, docent from episodes where config in ({cfgs})").df()
    e = meta.merge(eps, on="tid")
    e["setting"] = [
        HONEYPOT.get(env) if c == "aisi_honeypots" else evalcue_setting(env) if c == "evalcue" else SETTINGS[c]
        for c, env in zip(e.config, e.env)
    ]
    lab = [json.loads(x) if isinstance(x, str) else {} for x in e.labels]
    e["call"] = [d.get("call", 0) for d in lab]
    e["n_calls"] = [d.get("n_calls", 1) for d in lab]
    # inspect sources: one transcript per call. Prompt from call 0, generation from the last call of the episode.
    insp = e.config.str.startswith("inspect")
    last = e[insp].sort_values("call").groupby("episode_id").tail(1).set_index("episode_id")
    e = e[~insp | (e.call == 0)].copy()
    e["pend"] = [prompt_end(t, s) for t, s in zip(e.rendered_text, e.setting)]
    e = e[e.pend > 0]
    e["prompt_text"] = [t[:i] for t, i in zip(e.rendered_text, e.pend)]
    e["ph"] = [hashlib.md5(p.encode()).hexdigest()[:12] for p in e.prompt_text]
    gen = []
    for r in e.itertuples():
        if r.config.startswith("inspect"):
            lt = last.loc[r.episode_id] if r.episode_id in last.index else None
            t = lt.rendered_text if lt is not None else r.rendered_text
            gen.append(generated(t, t.find("<|assistant|>") if lt is not None and lt.call > 0 else prompt_end(t, r.setting)))
        else:
            gen.append(generated(r.rendered_text, r.pend))
    e["gen_text"] = gen

    # condition = what the experimenter varied (arm); inspect sources: the task's condition fields
    def cond(r: pd.Series, d: dict) -> str:
        if r.config == "inspect_agentic_misalignment":
            m = d.get("sample_metadata", {})
            return "|".join(str(m.get(k)) for k in ("scenario", "goal_type", "urgency_type", "prod", "test_eval_awareness"))
        if r.config.startswith("inspect"):
            return r.env
        return f"{r.env}|{r.arm}"

    labs = dict(zip(meta.tid, (json.loads(x) if isinstance(x, str) else {} for x in meta.labels)))
    e["cond"] = [cond(r, labs.get(r.tid, {})) for _, r in e.iterrows()]
    e["epoch"] = [labs.get(t, {}).get("epoch") for t in e.tid]

    # prompt-side read positions, pooled per prompt
    con.register("pe", e[["tid", "ph", "pend"]])
    reads = con.execute("""
        select pe.ph, p.tid, p.pos, p.token, p.role, p.kind, p.ctx_left40, p.ctx_right40, r.reader, r.sample, r.text
        from positions p join pe on p.tid = pe.tid join reads r on r.tid = p.tid and r.pos = p.pos
        where p.char_end >= 0 and p.char_end <= pe.pend and r.reader in ('glm_sft', 'glm_rl')""").df()
    # one source transcript per (ph, pos) for context; keep every reader text from every episode (engine noise = more samples)
    reads.to_parquet(a.out / "reads.parquet")
    npos = reads.groupby("ph").pos.nunique()
    own = reads.groupby("tid").pos.nunique()
    e["n_prompt_pos_own"] = e.tid.map(own).fillna(0).astype(int)
    e["n_prompt_pos_pooled"] = e.ph.map(npos).fillna(0).astype(int)
    prompts = e.drop_duplicates("ph")[["ph", "setting", "cond", "prompt_text", "tid"]].rename(columns={"tid": "example_tid"})
    prompts["n_pos"] = prompts.ph.map(npos).fillna(0).astype(int)
    prompts.to_parquet(a.out / "prompts.parquet")
    keep = ["tid", "episode_id", "config", "setting", "env", "arm", "item", "sample", "epoch", "cond", "ph", "aware", "positive",
            "run_dir", "row_index", "docent", "truncated", "n_prompt_pos_own", "n_prompt_pos_pooled", "gen_text"]  # fmt: skip
    e[keep].to_parquet(a.out / "episodes.parquet")
    s = e.groupby("setting").agg(
        eps=("tid", "size"),
        prompts=("ph", "nunique"),
        conds=("cond", "nunique"),
        with_reads=("n_prompt_pos_pooled", lambda x: (x > 0).sum()),
    )
    s["med_pos_per_prompt"] = prompts.groupby("setting").n_pos.median()
    print(s.to_string())
    print(f"{len(reads):,} prompt-side reader texts over {reads.groupby(['ph', 'pos']).ngroups:,} (prompt, position) pairs")


if __name__ == "__main__":
    main()
