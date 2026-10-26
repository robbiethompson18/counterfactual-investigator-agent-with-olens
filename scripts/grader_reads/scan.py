# /// script
# requires-python = ">=3.11"
# dependencies = ["pyarrow", "numpy"]
# ///
"""Scan every GLM-5.3 O-lens read in the expview imports for grader / eval / test / score / reward words.

One row per read of the RL GLM verbalizer (greedy), at every read position of every item:
dataset, item, position, section kind, turn, a hash of the token prefix (so reads of an identical
prefix, e.g. a tool schema shared by every arm, can be counted once), and per-family match bits:
raw (the read matches), novel (the match is absent from the 40 tokens before the position and the 16
after, i.e. not an echo of nearby text), item-novel (absent from the whole transcript), and the same
for the four T=1 samples (any sample). Matching reads are also written with their text and context.

    uv run --script scripts/grader_reads/scan.py --imports data/expview/imports --out data/runs/grader_reads/<run>
"""

import argparse
import bisect
import glob
import gzip
import hashlib
import json
import os
import re
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FAMILIES = {
    "grader": r"\b(?:auto)?grad(?:er|ers|ing|ed)\b",
    "eval": r"\beval(?:s|uat\w*)?\b",
    "test": r"\btest(?:s|ed|ing|er|ers)?\b",
    "score": r"\bscor(?:e|es|ed|ing)\b",
    "reward": r"\breward\w*",
    "benchmark": r"\b(?:benchmark\w*|rubric\w*)",
    "sim": r"\b(?:simulat\w*|fictional|hypothetical|role-?play\w*)",
}
FAM = list(FAMILIES)
RX = {k: re.compile(v, re.IGNORECASE) for k, v in FAMILIES.items()}
SECT = ["header", "system", "tool_schema", "user", "turn_start", "reasoning", "reply", "tool_call", "tool_result", "other"]
SID = {s: i for i, s in enumerate(SECT)}
GREEDY = "GLM verbalizer (RL)"
SAMPLES = "GLM verbalizer (RL) samples"


def bits(text, rx_ctx=None):
    b = 0
    for i, k in enumerate(FAM):
        if RX[k].search(text) and (rx_ctx is None or not RX[k].search(rx_ctx)):
            b |= 1 << i
    return b


def classify(tokens, sections):
    """Per section: (start, end, kind); reply tokens inside <tool_call>...</tool_call> become tool_call."""
    out = []
    for name, a, b in sections:
        n = name.lower()
        if n == "system":
            txt = "".join(tokens[a : min(b, a + 40)])
            kind = "tool_schema" if ("# Tools" in txt or "<tools>" in txt) else "system"
        elif n == "header":
            kind = "header"
        elif n == "user":
            kind = "user"
        elif n == "assistant":
            kind = "turn_start"
        elif n in ("reasoning", "think", "cot"):
            kind = "reasoning"
        elif n in ("reply", "answer"):
            kind = "reply"
        elif n in ("tool", "observation"):
            kind = "tool_result"
        else:
            kind = "other"
        if kind == "reply":
            inside, start = False, a
            for i in range(a, b):
                t = tokens[i]
                if not inside and "<tool_call>" in t:
                    if i > start:
                        out.append((start, i, "reply"))
                    inside, start = True, i
                elif inside and "</tool_call>" in t:
                    out.append((start, i + 1, "tool_call"))
                    inside, start = False, i + 1
            if b > start:
                out.append((start, b, "tool_call" if inside else "reply"))
        else:
            out.append((a, b, kind))
    return out


def scan_dataset(args):
    ds, run_dir, out = args
    with open(os.path.join(run_dir, "items.jsonl")) as fh:
        items = [json.loads(line) for line in fh]
    cols = {k: [] for k in ["item", "pos", "sect", "turn", "prefix", "raw", "novel", "inovel", "s_raw", "s_novel", "n_samples"]}
    item_rows, hits = [], []
    for ii, it in enumerate(items):
        tk = it.get("tokens")
        labels = it.get("labels") or {}
        item_rows.append(
            {
                "item": ii,
                "id": it["id"],
                "group": it.get("group"),
                "labels": json.dumps(labels),
                "extra": json.dumps(it.get("extra") or {})[:2000],
                "n_tokens": (tk or {}).get("n"),
            }
        )
        if not tk:
            continue
        p = os.path.join(run_dir, "tokens", tk["ref"] + ".json")
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            d = json.load(fh)
        toks = d["tokens"]
        secs = classify(toks, d.get("sections") or [])
        starts = [s[0] for s in secs]
        turn_of = []
        t = 0
        for s in secs:
            if s[2] == "turn_start":
                t += 1
            turn_of.append(t)
        full = "".join(toks)
        reads = d.get("reads") or {}
        positions = sorted(int(k) for k in reads)
        h = hashlib.blake2b(digest_size=8)
        j = 0
        for pos in positions:
            while j <= pos and j < len(toks):
                h.update(toks[j].encode("utf-8", "surrogatepass") + b"\x00")
                j += 1
            prefix = int.from_bytes(h.copy().digest(), "little", signed=True)
            g, smp = None, []
            for r in reads[str(pos)]:
                if r.get("tool") == GREEDY and r.get("lines"):
                    g = r["lines"][0][0]
                elif r.get("tool") == SAMPLES:
                    smp = [ln[0] for ln in r.get("lines") or []]
            if g is None:
                continue
            k = bisect.bisect_right(starts, pos) - 1
            sect = secs[k][2] if 0 <= k < len(secs) and pos < secs[k][1] else "other"
            turn = turn_of[k] if k >= 0 else 0
            ctx = "".join(toks[max(0, pos - 40) : pos + 17])
            raw = bits(g)
            novel = bits(g, ctx) if raw else 0
            inovel = bits(g, full) if raw else 0
            s_raw = s_novel = 0
            for s in smp:
                b = bits(s)
                if b:
                    s_raw |= b
                    s_novel |= bits(s, ctx)
            for c, v in zip(cols, [ii, pos, SID[sect], turn, prefix, raw, novel, inovel, s_raw, s_novel, len(smp)]):
                cols[c].append(v)
            if raw:
                hits.append(
                    {
                        "ds": ds,
                        "item": ii,
                        "id": it["id"],
                        "group": it.get("group"),
                        "pos": pos,
                        "sect": sect,
                        "turn": turn,
                        "prefix": prefix,
                        "raw": raw,
                        "novel": novel,
                        "inovel": inovel,
                        "read": g,
                        "before": "".join(toks[max(0, pos - 60) : pos + 1]),
                        "after": "".join(toks[pos + 1 : pos + 17]),
                    }
                )
    os.makedirs(out, exist_ok=True)
    safe = ds.replace("/", "__")
    tbl = pa.table(
        {
            "item": np.array(cols["item"], np.int32),
            "pos": np.array(cols["pos"], np.int32),
            "sect": np.array(cols["sect"], np.int8),
            "turn": np.array(cols["turn"], np.int16),
            "prefix": np.array(cols["prefix"], np.int64),
            "raw": np.array(cols["raw"], np.int16),
            "novel": np.array(cols["novel"], np.int16),
            "inovel": np.array(cols["inovel"], np.int16),
            "s_raw": np.array(cols["s_raw"], np.int16),
            "s_novel": np.array(cols["s_novel"], np.int16),
            "n_samples": np.array(cols["n_samples"], np.int8),
        }
    )
    pq.write_table(tbl, os.path.join(out, f"reads__{safe}.parquet"))
    pq.write_table(pa.Table.from_pylist(item_rows), os.path.join(out, f"items__{safe}.parquet"))
    with gzip.open(os.path.join(out, f"hits__{safe}.jsonl.gz"), "wt") as f:
        for hrow in hits:
            f.write(json.dumps(hrow) + "\n")
    return ds, len(items), len(cols["item"]), len(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imports", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma-separated substrings of dataset ids")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    jobs = []
    for rj in sorted(glob.glob(os.path.join(a.imports, "**/run.json"), recursive=True)):
        d = os.path.dirname(rj)
        if not os.path.exists(os.path.join(d, "items.jsonl")):
            continue
        ds = os.path.relpath(d, a.imports)
        if a.only and not any(s in ds for s in a.only.split(",")):
            continue
        jobs.append((ds, os.path.realpath(d), a.out))
    # biggest first so the pool finishes together
    jobs.sort(key=lambda j: -sum(e.stat().st_size for e in os.scandir(os.path.join(j[1], "tokens")) if e.is_file()))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "meta.json"), "w") as fh:
        json.dump({"families": FAMILIES, "sections": SECT, "datasets": [j[0] for j in jobs]}, fh, indent=1)
    with Pool(a.workers) as p:
        for r in p.imap_unordered(scan_dataset, jobs):
            print(*r, flush=True)


if __name__ == "__main__":
    main()
