# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2", "pyarrow>=17", "tokenizers>=0.20", "huggingface_hub>=0.25"]
# [tool.ty.environment]
# extra-paths = ["."]
# [tool.ty.analysis]
# allowed-unresolved-imports = ["tokenizers"]
# ///
"""Pick the read positions the judges see, from scan.py's hits, and write one shard file per judge.

    uv run --script scripts/glm_wild_grader/candidates.py RUN [--ds data/olens_search/ds_wild]

A candidate is one read position (deduped by token prefix) whose bullet has a family keyword that is NOT in the 60 tokens
before / 16 after. Pools (`pool` column): every `sit` read (bullet also talks about the task, the AI/model/user, being
tested, my score, ...) in wildchat and rlvr1; a seeded sample of the other novel wildchat reads, to measure what the `sit`
filter misses; a seeded sample of `sit` reads in swesmith and tmax (thousands each). Writes RUN/judge/candidates.parquet,
RUN/judge/shards/<shard>.md (what each judge reads) and RUN/judge/transcripts/<tid>.txt (full renders for the judges to grep).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).parent))
from scan import RX, SLOTS, token_texts

SIT = re.compile(
    r"\b(?:this|these|such|typical|current|the) (?:\w+[- ])?(?:tasks?|scenarios?|benchmarks?|evals?|evaluations?|environments?|tests?|setups?|"  # noqa: E501
    r"challenges?|exercises?|episodes?|prompts?|questions?|conversations?|requests?)\b(?! (?:file|case|suite|script|data|set|function|command|"  # noqa: E501
    r"runner|output|harness)s?)|\b(?:the|an?) (?:ai|agent|model|assistant|llm|chatbot|bot|language model)\b|\b(?:the )?user\b|"
    r"\btest(?:ing|s)? (?:me|whether|if|you)\b|\bscenario\b|\btest of my\b|\bmy (?:\w+ )?(?:score|reward|grade|answer|response|performance)s?\b"  # noqa: E501
    r"|\bmaximi[sz]\w* (?:the )?(?:score|reward)|\b(?:i|we) (?:am|are|'m|'re) being\b|\bbeing (?:tested|evaluated|graded|scored|judged|monitored)\b"  # noqa: E501
    r"|\b(?:the|an?|our) (?:auto)?grader\b|\bthe (?:evaluator|rubric|benchmark)\b",
    re.IGNORECASE,
)
SAMPLE = {"wildchat_other": 400, "swesmith": 250, "tmax": 250}
SHARDS = {"wildchat": 4, "rlvr1": 2, "swesmith": 1, "tmax": 1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--ds", type=Path, default=Path("data/olens_search/ds_wild"))
    a = ap.parse_args()
    hits = pd.read_parquet(a.run / "scan/hits.parquet")
    hits = hits[~hits.kw_in_window].assign(sit=lambda d: d.bullet.str.contains(SIT))
    # one row per read position (prefix): all keyword families and slots that matched there
    by = hits.groupby("prefix")
    reads = by.first()[["family", "tid", "role", "pos", "token", "thinking", "dense"]].assign(
        kws=by.kw.agg(lambda s: sorted(set(s))),
        slots=by.slot.agg(lambda s: sorted(set(s))),
        sit=by.sit.any(),
        kw_in_transcript=by.kw_in_transcript.all(),
    )
    reads = reads.reset_index()
    picks = []
    for fam, g in reads.groupby("family"):
        if fam in ("wildchat", "rlvr1"):
            picks.append(g[g.sit].assign(pool=f"{fam}_sit"))
        if fam == "wildchat":
            o = g[~g.sit]
            picks.append(o.sample(min(len(o), SAMPLE["wildchat_other"]), random_state=0).assign(pool="wildchat_other_sample"))
        if fam in ("swesmith", "tmax"):
            s = g[g.sit]
            picks.append(s.sample(min(len(s), SAMPLE[fam]), random_state=0).assign(pool=f"{fam}_sit_sample"))
    cand = pd.concat(picks, ignore_index=True).sort_values(["family", "pool", "tid", "pos"], ignore_index=True)
    pool_sizes = {
        "wildchat_sit": int(reads[(reads.family == "wildchat") & reads.sit].shape[0]),
        "wildchat_other": int(reads[(reads.family == "wildchat") & ~reads.sit].shape[0]),
        "rlvr1_sit": int(reads[(reads.family == "rlvr1") & reads.sit].shape[0]),
        "swesmith_sit": int(reads[(reads.family == "swesmith") & reads.sit].shape[0]),
        "tmax_sit": int(reads[(reads.family == "tmax") & reads.sit].shape[0]),
        **{f"{f}_novel_reads": int((reads.family == f).sum()) for f in reads.family.unique()},
    }
    print(pool_sizes)

    # the five RL bullets at each candidate position, and its context from the token ids
    want = set(zip(cand.tid, cand.pos))
    cols = ["tid", "pos", *SLOTS, *[f"fve_{s}" for s in SLOTS]]
    capped = pd.read_parquet(a.ds / "olens_wild_rollouts.parquet", columns=["tid", "pos", *SLOTS])
    capped = capped.merge(
        pd.read_parquet(a.ds / "fve_readouts.parquet", columns=[c for c in cols if c not in SLOTS]).drop_duplicates(["tid", "pos"]),
        on=["tid", "pos"],
        how="left",
    )
    dense = pd.read_parquet(a.ds / "wild/readouts_dense.parquet", columns=cols, filters=[("tid", "in", sorted({t for t, _ in want}))])
    allr = pd.concat([capped, dense]).drop_duplicates(["tid", "pos"])
    allr = allr[[k in want for k in zip(allr.tid, allr.pos)]]
    cand = cand.merge(allr, on=["tid", "pos"], how="left")

    tx = pd.read_parquet(a.ds / "transcripts.parquet", columns=["tid", "ids", "task_id"]).drop_duplicates("tid").set_index("tid")
    tok = Tokenizer.from_file(hf_hub_download("zai-org/GLM-5.3", "tokenizer.json"))
    tdir = a.run / "judge/transcripts"
    tdir.mkdir(parents=True, exist_ok=True)
    before, after = [], []
    for tid, g in cand.groupby("tid", sort=False):
        tt = token_texts(tok, list(tx.at[tid, "ids"]))
        (tdir / f"{tid}.txt").write_text("".join(tt))
        for i, r in g.iterrows():
            before.append((i, "".join(tt[max(0, r.pos - 150) : r.pos + 1])[-1200:]))
            after.append((i, "".join(tt[r.pos + 1 : r.pos + 41])))
    cand["before"] = pd.Series(dict(before))
    cand["after"] = pd.Series(dict(after))
    cand["cid"] = [f"{p[:2]}{i:04d}" for i, p in zip(range(len(cand)), cand.family)]

    sdir = a.run / "judge/shards"
    sdir.mkdir(parents=True, exist_ok=True)
    shard_col = []
    for fam, g in cand.groupby("family", sort=False):
        n = -(-len(g) // SHARDS[fam])
        for k in range(SHARDS[fam]):
            name = f"{fam}_{k}"
            part = g.iloc[k * n : (k + 1) * n].sample(frac=1, random_state=k)  # pools mixed so a judge can't infer one from position
            shard_col += [(i, name) for i in part.index]
            (sdir / f"{name}.md").write_text("\n\n".join(block(r) for r in part.itertuples()))
    cand["shard"] = pd.Series(dict(shard_col))
    cand.to_parquet(a.run / "judge/candidates.parquet")
    print(cand.groupby(["shard"]).size().to_string())
    (a.run / "judge/pool_sizes.txt").write_text(f"{pool_sizes}\n")


def block(r) -> str:
    """Exactly what a judge sees for one candidate (also stored as the judge call's input in expview)."""
    lines = []
    for s in SLOTS:
        b = getattr(r, s) or ""
        mark = " ← keyword" if any(RX[k].search(b) for k in r.kws) else ""
        fve = getattr(r, f"fve_{s}")
        lines.append(f"- {s} (FVE {fve:.2f}){mark}: {b.strip()}" if pd.notna(fve) else f"- {s}: {b.strip()}{mark}")
    return (
        f"### {r.cid}\n"
        f"family: {r.family} · transcript: {r.tid} (judge/transcripts/{r.tid}.txt) · role at read: {r.role} · token index {r.pos}\n"
        f"keyword families matched: {', '.join(r.kws)} · keyword family appears anywhere in the transcript text: {'yes' if r.kw_in_transcript else 'no'}\n\n"  # noqa: E501
        f"Transcript up to and including the read token (last ~150 tokens):\n```\n{r.before}\n```\n"
        f"What actually came next (40 tokens):\n```\n{r.after}\n```\n"
        f"O-lens reads at this token (RL lens, greedy + 4 samples at T=1):\n" + "\n".join(lines)
    )


if __name__ == "__main__":
    main()
