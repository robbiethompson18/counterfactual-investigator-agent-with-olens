# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2", "pyarrow>=17", "tokenizers>=0.20", "huggingface_hub>=0.25"]
# [tool.ty.environment]
# extra-paths = [".", "../../expview", "../../expview/importers"]
# [tool.ty.analysis]
# allowed-unresolved-imports = ["tokenizers"]
# ///
"""Merge the subagent verdicts into the candidates, print the counts, and write the expview run.

    uv run --script scripts/glm_wild_grader/export.py RUN [--ds data/olens_search/ds_wild]

RUN/judge/verdicts/*.jsonl (one line per candidate) -> RUN/judged.parquet, RUN/summary.json and RUN/expview/: one item per
judged transcript, one judge call per candidate (exactly the instructions + candidate block the subagent read, and its
JSON verdict). Transcripts with an own_situation or ambiguous verdict open on their tokens with the five RL bullets
inline at every candidate position; content-only transcripts carry the judge calls only (keeps the run small).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "expview"))
sys.path.insert(0, str(HERE.parents[1] / "expview/importers"))
from candidates import block
from celeste import sections_of
from scan import SLOTS, token_texts

from expview import Run, call, link, read

JUDGE_MODEL = "claude-opus-5-5 (Claude Code subagent, one per shard)"
READER = "GLM verbalizer (RL), greedy + 4 samples"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--ds", type=Path, default=Path("data/olens_search/ds_wild"))
    a = ap.parse_args()
    cand = pd.read_parquet(a.run / "judge/candidates.parquet")
    v = pd.concat([pd.read_json(f, lines=True) for f in sorted((a.run / "judge/verdicts").glob("*.jsonl"))]).drop_duplicates(
        "cid", keep="last"
    )
    v = v.rename(columns={"genuine": "genuine_judge"})
    df = cand.merge(v, on="cid", how="left", validate="1:1")
    missing = df.category.isna().sum()
    print(f"{len(df)} candidates, {missing} without a verdict")
    df["genuine"] = df.category.eq("own_situation")
    df["verbalised"] = df.verbalised.fillna(False).astype(bool)
    df["unverbalised_genuine"] = df.genuine & ~df.verbalised
    df.drop(columns=["before", "after"]).to_parquet(a.run / "judged.parquet")

    pools = {}
    for pool, g in df.groupby("pool"):
        pools[pool] = {
            "candidates": len(g),
            "own_situation": int(g.genuine.sum()),
            "ambiguous": int(g.category.eq("ambiguous").sum()),
            "unverbalised_own": int(g.unverbalised_genuine.sum()),
            "verbalised_by": g[g.genuine].verbalised_by.value_counts().to_dict(),
            "transcripts": int(g.tid.nunique()),
            "transcripts_own": int(g[g.genuine].tid.nunique()),
            "transcripts_unverbalised_own": int(g[g.unverbalised_genuine].tid.nunique()),
        }
    (a.run / "summary.json").write_text(json.dumps({"pools": pools, "pool_sizes": (a.run / "judge/pool_sizes.txt").read_text()}, indent=1))
    print(json.dumps(pools, indent=1))

    instructions = (HERE / "judge_instructions.md").read_text()
    tx = (
        pd.read_parquet(a.ds / "transcripts.parquet", columns=["tid", "ids", "task_id", "thinking"]).drop_duplicates("tid").set_index("tid")
    )
    tok = Tokenizer.from_file(hf_hub_download("zai-org/GLM-5.3", "tokenizer.json"))
    run = Run(
        a.run / "expview",
        title="GLM-5.3 wild rollouts: grader / eval awareness in O-lens reads",
        description=(
            "Every RL O-lens read (greedy + 4 samples) of [ceselder/glm-5.3-olens-wild-rollouts](https://huggingface.co/datasets/"
            "ceselder/glm-5.3-olens-wild-rollouts) scanned for grader/eval/test/score/reward/monitor/judge/benchmark/assess words "
            "absent from the 60 tokens before and 16 after; candidates judged by Claude subagents. One item per judged transcript. "
            "`genuine` = some read there is about the model's OWN situation being graded/tested/evaluated; `unverbalised` = such a "
            "read whose content the transcript never states. Each `judge` call is exactly the instructions and candidate block the "
            "subagent read, and its JSON verdict (subagents could also grep the full transcript). Transcripts: WildChat-1M prompts "
            "(ODC-BY, Zhao et al. 2024) answered on-policy by GLM-5.3; open-athena GLM-5.3 tmax/swesmith/RLVR1 traces. "
            "Write-up: docs/glm-wild-grader-awareness.md."
        ),
        label_order=["unverbalised_genuine", "genuine", "verbalised", "family", "keyword", "pool"],
        readers={
            READER: {
                "about": "Celeste's GLM-5.3 RL one-bullet O-lens (`rl1b_2/step_0100`) at block 51; line score = per-bullet FVE. "
                "Read label = the judge's category for that position.",
                "color": "#2f5d8a",
            }
        },
    )
    run.reset()
    for tid, g in df.groupby("tid"):
        g = g.sort_values("pos")
        rows: list[Any] = list(g.itertuples())
        gen = g[g.genuine]
        calls = [
            call(
                f"judge {r.cid} @{r.pos}",
                JUDGE_MODEL,
                messages=[{"role": "system", "content": instructions}, {"role": "user", "content": block(r)}],
                output={"role": "assistant", "content": json.dumps({k: getattr(r, k) for k in VKEYS}, default=str)},
                parsed={k: getattr(r, k) for k in ("category", "verbalised", "verbalised_by", "confidence")},
                token_refs={str(r.pos): int(r.pos)},
            )
            for r in rows
        ]
        tokens = None
        if g.category.isin(["own_situation", "ambiguous"]).any():
            toks = token_texts(tok, list(tx.at[tid, "ids"]))
            reads = {
                str(r.pos): [
                    read(
                        READER,
                        [
                            [getattr(r, s) or "", round(float(getattr(r, f"fve_{s}")), 3)]
                            if pd.notna(getattr(r, f"fve_{s}"))
                            else [getattr(r, s) or ""]
                            for s in SLOTS
                        ],
                        label=f"{r.category}{' · unverbalised' if r.unverbalised_genuine else ''} ({r.cid}; {', '.join(r.kws)})",
                    )
                ]
                for r in rows
            }
            tokens = {
                "tokens": toks,
                "sections": sections_of(toks),
                "reads": reads,
                "legend": "Underlined tokens are judged candidate reads; the side panel has each judge call.",
            }
        kws = sorted({k for ks in (gen if len(gen) else g).kws for k in ks})
        run.add(
            id=tid,
            group=g.family.iloc[0],
            labels={
                "unverbalised_genuine": bool(g.unverbalised_genuine.any()),
                "genuine": bool(len(gen)),
                "verbalised": bool(gen.verbalised.any()) if len(gen) else None,
                "family": g.family.iloc[0],
                "keyword": ",".join(kws),
                "pool": ",".join(sorted(set(g.pool))),
                "thinking": tx.at[tid, "thinking"],
            },
            calls=calls,
            tokens=tokens,
            extra={"task_id": tx.at[tid, "task_id"], "candidates": g.cid.tolist()},
        )
    print(link(run.path, item=df[df.unverbalised_genuine].tid.iloc[0]) if df.unverbalised_genuine.any() else link(run.path))


VKEYS = ["cid", "category", "genuine_judge", "verbalised", "verbalised_by", "verbalised_quote", "rationale", "confidence", "key_bullet"]

if __name__ == "__main__":
    main()
