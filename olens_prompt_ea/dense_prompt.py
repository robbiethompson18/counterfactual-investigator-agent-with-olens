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
SOURCES = ["ds/agentic_misalignment", "ds/aisi_honeypots", "ds/evalcue", "ds_new/cfi_donation_bet"]


def load(path: Path, cols: list[str]) -> pd.DataFrame:
    have = set(pq.ParquetFile(path).schema_arrow.names)
    return pd.read_parquet(path, columns=[c for c in cols if c in have])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--ds", type=Path, required=True)
    a = ap.parse_args()
    out = []
    for src in SOURCES:
        d = a.ds / src
        parts = []
        for f in ("readouts_dense.parquet", "readouts.parquet"):
            if (d / f).exists():
                parts.append(load(d / f, ["tid", "pos", "role", "msg_index", *RL]))
        r = pd.concat(parts, ignore_index=True).drop_duplicates(["tid", "pos"])
        gen = r[r.role.isin(["assistant", "reasoning", "tool_call"])].groupby("tid").msg_index.min()
        r["first_gen"] = r.tid.map(gen)
        prompt = r[r.role.isin(["system", "user"]) & (r.msg_index < r.first_gen)].copy()
        prompt["hit"] = prompt[RL].fillna("").apply(lambda row: any(RX.search(x) for x in row), axis=1)
        per = prompt[["tid", "pos", "hit"]]
        if (d / "dense_prefix_dups.parquet").exists():
            dup = pd.read_parquet(d / "dense_prefix_dups.parquet", columns=["tid", "pos", "canon_tid", "canon_pos"])
            canon = r[r.role.isin(["system", "user"])][["tid", "pos"]].copy()
            canon["hit"] = r.loc[canon.index, RL].fillna("").apply(lambda row: any(RX.search(x) for x in row), axis=1)
            m = dup.merge(canon.rename(columns={"tid": "canon_tid", "pos": "canon_pos"}), on=["canon_tid", "canon_pos"])
            m = m[m.tid.map(gen).notna()]
            # a duplicate position is prompt-side for this episode when it sits before its own first generated token,
            # approximated by: the canonical read is a system/user token (only shared prefixes are deduplicated)
            per = pd.concat([per, m[["tid", "pos", "hit"]]], ignore_index=True).drop_duplicates(["tid", "pos"])
        g = per.groupby("tid").agg(dense_prompt_positions=("pos", "nunique"), dense_prompt_any=("hit", "any"))
        g["source"] = src
        out.append(g.reset_index())
        med, rate = g.dense_prompt_positions.median(), g.dense_prompt_any.mean()
        print(f"{src}: {len(g):,} episodes, median prompt positions {med:.0f}, any-hit {rate:.2f}")
    pd.concat(out).to_parquet(a.run / "dense_prompt.parquet")


if __name__ == "__main__":
    main()
