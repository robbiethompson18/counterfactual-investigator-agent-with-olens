# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2", "pyarrow>=17"]
# ///
"""Prompt-side RL O-lens any-hit per episode from Celeste's DENSE (every-token) readouts, which landed after the
olens_search DB snapshot this study first used (2026-10-24 18:32 UTC): ds/agentic_misalignment, ds/aisi_honeypots,
ds/evalcue, ds_new/cfi_donation_bet (readouts_dense.parquet, 2026-10-24 21:47 - 10-25 00:26 UTC).

Per episode (tid), prompt-side positions = system / user tokens before the first generated message (msg_index below the
episode's first assistant / reasoning / tool_call row), collected from three places:
  readouts_dense.parquet   the episode's own dense reads
  readouts.parquet         the sparse anchored reads
  dense_prefix_dups.parquet  positions not re-read because an identical prefix was read in another transcript: mapped to
                           (canon_tid, canon_pos), whose reads are used
Hit = any RL read (greedy + 4 samples) matches simulation|eval|scenario (case-insensitive).

    uv run --script olens_prompt_ea/dense_prompt.py RUN --ds data/olens_search   -> RUN/dense_prompt.parquet
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

RX = re.compile("simulation|eval|scenario", re.IGNORECASE)
RL = ["rl_greedy", "rl_s1", "rl_s2", "rl_s3", "rl_s4"]
WORDS = {
    "sim": re.compile(r"\bsimulation", re.IGNORECASE),
    "eval": re.compile(r"\beval", re.IGNORECASE),
    "scen": re.compile(r"\bscenario", re.IGNORECASE),
}


def flags(df: pd.DataFrame) -> pd.DataFrame:
    """Per row: does any RL read match each word, and any of them."""
    txt = df[RL].fillna("").agg(" || ".join, axis=1)
    out = pd.DataFrame({f"hit_{k}": txt.str.contains(rx) for k, rx in WORDS.items()}, index=df.index)
    out["hit"] = out.any(axis=1)
    return out


SOURCES = ["ds/agentic_misalignment", "ds/aisi_honeypots", "ds/evalcue", "ds_new/cfi_donation_bet"]


def load(path: Path, cols: list[str]) -> pd.DataFrame:
    have = set(pq.ParquetFile(path).schema_arrow.names)
    return pd.read_parquet(path, columns=[c for c in cols if c in have])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--ds", type=Path, required=True)
    ap.add_argument("--punct-only", action="store_true", help="keep only punctuation / newline token positions")
    ap.add_argument("--out", default="dense_prompt.parquet")
    ap.add_argument("--bullets", action="store_true", help="also write RUN/bullets_<source>.parquet (prompt-side RL bullets)")
    a = ap.parse_args()
    out = []
    for src in SOURCES:
        d = a.ds / src
        parts = []
        for f in ("readouts_dense.parquet", "readouts.parquet"):
            if (d / f).exists():
                x = load(d / f, ["tid", "pos", "role", "msg_index", "is_punct", "token", "left_ctx", "ctx_left40", *RL])
                if "left_ctx" not in x and "ctx_left40" in x:
                    x = x.rename(columns={"ctx_left40": "left_ctx"})
                if "is_punct" not in x:
                    x["is_punct"] = True  # the sparse anchored reads are punctuation / newline tokens only
                parts.append(x)
        r = pd.concat(parts, ignore_index=True).drop_duplicates(["tid", "pos"])
        gen_all = r[r.role.isin(["assistant", "reasoning", "tool_call"])].groupby("tid").msg_index.min()
        if a.punct_only:  # Celeste's rule: the workspace lives at punctuation; drop mid-word / every-token dense reads
            r = r[r.is_punct.fillna(False).astype(bool)]
        gen = gen_all
        r["first_gen"] = r.tid.map(gen)
        prompt = r[r.role.isin(["system", "user"]) & (r.msg_index < r.first_gen)].copy()
        prompt = prompt.join(flags(prompt))
        if a.bullets:  # the RL bullets themselves, for a judge: own prompt-side reads + canonical reads of deduplicated prefixes
            keep = ["tid", "pos", "token", "left_ctx", *RL]
            own = prompt[keep]
            if (d / "dense_prefix_dups.parquet").exists():
                dup = pd.read_parquet(d / "dense_prefix_dups.parquet", columns=["tid", "pos", "canon_tid", "canon_pos"])
                su = r[r.role.isin(["system", "user"])][keep].rename(columns={"tid": "canon_tid", "pos": "canon_pos"})
                mm = dup.merge(su, on=["canon_tid", "canon_pos"])
                mm = mm[mm.tid.map(gen).notna()][keep]
                own = pd.concat([own, mm], ignore_index=True).drop_duplicates(["tid", "pos"])
            own.assign(source=src).to_parquet(a.run / f"bullets_{src.replace('/', '_')}.parquet")
        cols = ["hit", "hit_sim", "hit_eval", "hit_scen"]
        per = prompt[["tid", "pos", *cols]]
        if (d / "dense_prefix_dups.parquet").exists():
            dup = pd.read_parquet(d / "dense_prefix_dups.parquet", columns=["tid", "pos", "canon_tid", "canon_pos"])
            su = r[r.role.isin(["system", "user"])]
            canon = su[["tid", "pos"]].join(flags(su))
            m = dup.merge(canon.rename(columns={"tid": "canon_tid", "pos": "canon_pos"}), on=["canon_tid", "canon_pos"])
            m = m[m.tid.map(gen).notna()]
            # a duplicate position is prompt-side for this episode when it sits before its own first generated token,
            # approximated by: the canonical read is a system/user token (only shared prefixes are deduplicated)
            per = pd.concat([per, m[["tid", "pos", *cols]]], ignore_index=True).drop_duplicates(["tid", "pos"])
        g = per.groupby("tid").agg(
            dense_prompt_positions=("pos", "nunique"), dense_prompt_any=("hit", "any"),
            dense_any_sim=("hit_sim", "any"), dense_any_eval=("hit_eval", "any"), dense_any_scen=("hit_scen", "any"),
        )  # fmt: skip
        g["source"] = src
        out.append(g.reset_index())
        med, rate = g.dense_prompt_positions.median(), g.dense_prompt_any.mean()
        print(f"{src}: {len(g):,} episodes, median prompt positions {med:.0f}, any-hit {rate:.2f}")
    pd.concat(out).to_parquet(a.run / a.out)


if __name__ == "__main__":
    main()
