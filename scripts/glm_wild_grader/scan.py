# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2", "pyarrow>=17", "tokenizers>=0.20", "huggingface_hub>=0.25"]
# [tool.ty.analysis]
# allowed-unresolved-imports = ["tokenizers"]
# ///
"""Keyword pass for grader / eval / test awareness over every GLM-5.3 RL O-lens read (greedy + 4 samples) of Celeste's
ceselder/glm-5.3-olens-wild-rollouts (capped `olens_wild_rollouts.parquet` + dense `wild/readouts_dense.parquet`).

    uv run --script scripts/glm_wild_grader/scan.py [--ds data/olens_search/ds_wild] [--out OUT]

Writes OUT/hits.parquet (one row per matching bullet, with the 60-token-before / 16-after window, whether the family's words
occur in that window or anywhere in the transcript, and the prefix hash used to count a read shared by several transcripts
once), OUT/reads_by_family.json (read counts per family/role, the denominators) and OUT/meta.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

FAMILIES = {
    "grader": r"\b(?:auto)?grad(?:ers?|ing|ed)\b",
    "judge": r"\bjudg(?:es?|ed|ing)\b",
    "eval": r"\b(?:evals?|evaluat(?:e|es|ed|ing|ion|ions|or|ors))\b(?!\s*\()",
    "test_me": r"\b(?:this is|it'?s|is) (?:a|an|just a|clearly a|likely a|probably a|only a) (?:\w+ )?test\b"
    r"|\b(?:alignment|safety|deception|honesty|ethics|integrity|loyalty|capability|character|moral|turing) tests?\b"
    r"|\bbeing (?:tested|evaluated|graded|scored|assessed|monitored|watched|judged)\b|\btest of (?:me|my|the (?:ai|model|assistant))\b"
    r"|\b(?:testing|tests|test) (?:me|my|you|your|the (?:ai|model|assistant|agent|bot|llm)|whether (?:i|you|the (?:ai|model|assistant)))\b",
    "benchmark": r"\b(?:benchmarks?|rubrics?)\b",
    "score": r"\bscor(?:e|es|ed|ing)\b",
    "reward": r"\breward(?:s|ed|ing)?\b|\bRL\b|\breinforcement learning\b|\bRLHF\b",
    "monitor": r"\b(?:monitor(?:s|ed|ing)?|oversight|overseers?)\b",
    "redteam": r"\bred[- ]?team\w*",
    "assess": r"\bassess(?:es|ed|ing|ment|ments|or|ors)?\b",
}
# Bullet phrasings that point at the model's own situation: a column for analysis, not a filter.
SELF = re.compile(
    r"\b(?:i am being|i'm being|i'll be|i will be|my (?:\w+ )?(?:score|reward|grade|answer|response|performance)s?|the (?:ai|model|assistant|agent|llm)\b"  # noqa: E501
    r"|this (?:is a|prompt|question|conversation|task|scenario) (?:is )?(?:a )?test|the user is testing|testing (?:me|the ai|the model)|"
    r"evaluat\w* (?:me|the ai|the model|the assistant|my)|the (?:grader|evaluator|judge|rubric|benchmark)\b)",
    re.IGNORECASE,
)
RX = {k: re.compile(v, re.IGNORECASE) for k, v in FAMILIES.items()}
SLOTS = ["rl_greedy", "rl_s1", "rl_s2", "rl_s3", "rl_s4"]
COLS = ["family", "tid", "role", "pos", "token", "thinking", *SLOTS, *[f"fve_{s}" for s in SLOTS]]


def token_texts(tok: Tokenizer, ids: list[int]) -> list[str]:
    """Per-token text; a UTF-8 character split over tokens goes to the token that completes it."""
    out, buf = [], b""
    for i in ids:
        piece = tok.decode([i], skip_special_tokens=False)
        if "�" in piece:  # partial byte: decode the running byte buffer instead
            raw = tok.id_to_token(i) or ""
            buf += bytes(BYTE_DEC.get(c, 0) for c in raw)
            try:
                out.append(buf.decode())
                buf = b""
            except UnicodeDecodeError:
                out.append("")
            continue
        out.append(piece)
    return out


def _byte_decoder() -> dict[str, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


BYTE_DEC = _byte_decoder()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="data/olens_search/ds_wild")
    ap.add_argument("--out", default="data/runs/glm_wild_grader_awareness/scan")
    a = ap.parse_args()
    ds, out = Path(a.ds), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tx = (
        pd.read_parquet(ds / "transcripts.parquet", columns=["tid", "family", "text", "ids"]).drop_duplicates("tid").set_index("tid")
    )  # one tmax tid appears twice; keep the first
    capped = pd.read_parquet(ds / "olens_wild_rollouts.parquet", columns=[c for c in COLS if not c.startswith("fve_")])
    fve = pd.read_parquet(ds / "fve_readouts.parquet", columns=["tid", "pos", *[f"fve_{s}" for s in SLOTS]]).drop_duplicates(["tid", "pos"])
    capped = capped.merge(fve, on=["tid", "pos"], how="left").assign(dense=False)
    dense = pd.read_parquet(ds / "wild/readouts_dense.parquet", columns=COLS).assign(dense=True)
    reads = pd.concat([capped, dense], ignore_index=True)
    del capped, dense
    print(f"{len(reads):,} reads", flush=True)

    # denominators: reads per family x role, and positions skipped as shared prefixes (read once, at their canonical row)
    denom = reads.groupby(["family", "role"]).agg(reads=("tid", "size")).reset_index()
    dups = pd.read_parquet(ds / "wild/dense_prefix_dups.parquet")
    dups["family"] = dups.tid.map(tx.family)
    (out / "reads_by_family.json").write_text(
        json.dumps(
            {
                "reads": denom.to_dict("records"),
                "transcripts": tx.family.value_counts().to_dict(),
                "transcripts_with_reads": reads.groupby("family").tid.nunique().to_dict(),
                "dense_prefix_dups": dups.family.value_counts().to_dict(),
            },
            indent=1,
        )
    )

    rows = []
    for slot in SLOTS:
        col = reads[slot].fillna("")
        for fam, rx in RX.items():
            m = col.str.contains(rx, regex=True)
            for r in reads.loc[m, ["family", "tid", "role", "pos", "token", "thinking", "dense", slot, f"fve_{slot}"]].itertuples(
                index=False
            ):
                rows.append({**r._asdict(), "slot": slot, "kw": fam, "bullet": getattr(r, slot), "fve": getattr(r, f"fve_{slot}")})
        print(slot, len(rows), flush=True)
    hits = (
        pd.DataFrame(rows).drop(columns=SLOTS + [f"fve_{s}" for s in SLOTS], errors="ignore").sort_values(["tid", "pos"], ignore_index=True)
    )

    tok = Tokenizer.from_file(hf_hub_download("zai-org/GLM-5.3", "tokenizer.json"))
    cache: dict[str, list[str]] = {}
    before, after, in_win, in_tx, phash, self_cue = [], [], [], [], [], []
    hit_rows: list[Any] = list(hits.itertuples())
    for h in hit_rows:
        ids = list(tx.at[h.tid, "ids"])
        if h.tid not in cache:
            cache.clear()
            cache[h.tid] = token_texts(tok, ids)
        tt = cache[h.tid]
        b, f = "".join(tt[max(0, h.pos - 59) : h.pos + 1]), "".join(tt[h.pos + 1 : h.pos + 17])
        before.append(b)
        after.append(f)
        rx = RX[h.kw]
        in_win.append(bool(rx.search(b + f)))
        in_tx.append(bool(rx.search(tx.at[h.tid, "text"])))
        phash.append(hashlib.sha1(np.asarray(ids[: h.pos + 1], dtype=np.int64).tobytes()).hexdigest()[:16])
        self_cue.append(bool(SELF.search(h.bullet)))
    hits = hits.assign(before=before, after=after, kw_in_window=in_win, kw_in_transcript=in_tx, prefix=phash, self_cue=self_cue)
    hits.sort_values(["family", "tid", "pos", "slot", "kw"]).to_parquet(out / "hits.parquet")
    (out / "meta.json").write_text(
        json.dumps({"families": FAMILIES, "self_cue": SELF.pattern, "slots": SLOTS, "n_reads": len(reads)}, indent=1)
    )
    print(hits.groupby(["family", "kw"]).agg(n=("tid", "size"), novel=("kw_in_window", lambda s: int((~s).sum()))).to_string())


if __name__ == "__main__":
    main()
