"""DEAD END (2026-10-23), never merged: residual reads surface no more new content than a norm-matched control.
Kept on a closed branch for reference only; results in docs/olens-residual-reads.md on main.

Summary table for scripts/olens_residual_reads.py output (every reads*.jsonl in RUN_DIR): one row per (layer, arm, beta, gamma, mu_scale).

    uv run python scripts/olens_residual_summary.py RUN_DIR

Columns: mean joint_fve (bullets vs the real z), resid_fve (vs the residual), gain_fve (beyond the continuation atoms),
cos_in (raw cos of h' with h), same1 (share of reads whose first bullet equals the plain read's), and three word-overlap
shares over the bullets' content words (lowercased, >= 4 letters): in the 32 tokens after p (fut32), anywhere in the
transcript (any), and in none of it (novel). `novel` is the number the residual idea needs to move without the reads
turning to noise.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

WORD = re.compile(r"[a-z][a-z']{3,}")


def words(s):
    return set(WORD.findall(s.lower()))


def transcript_text(row):
    parts = []
    for m in row["messages"]:
        for k in ("content", "reasoning_content"):
            if isinstance(m.get(k), str):
                parts.append(m[k])
        for tc in m.get("tool_calls") or []:
            parts.append(json.dumps(tc["function"]["arguments"]))
    return " ".join(parts)


def main():
    run = Path(sys.argv[1])
    full = {r["id"]: words(transcript_text(r)) for r in map(json.loads, (run / "requests.jsonl").open())}
    reads = [json.loads(line) for f in sorted(run.glob("reads*.jsonl")) for line in f.open()]  # reads.jsonl + follow-ups
    plain = {(r["id"], r["token_index"], r["layer"]): r["bullets"][0]["text"] if r["bullets"] else ""
             for r in reads if (r["arm"], r["beta"], r["gamma"], r["mu_scale"]) == ("main", 0.0, 1.0, 1.0)}  # fmt: skip
    agg = defaultdict(lambda: defaultdict(list))
    for r in reads:
        a = agg[(r["layer"], r["arm"], r["beta"], r["gamma"], r["mu_scale"])]
        for k in ("joint_fve", "resid_fve", "gain_fve", "cos_in", "cont_fve"):
            a[k].append(r[k])
        first = r["bullets"][0]["text"] if r["bullets"] else ""
        a["same1"].append(first == plain.get((r["id"], r["token_index"], r["layer"])))
        w = set().union(*(words(b["text"]) for b in r["bullets"])) if r["bullets"] else set()
        if w:
            fut = words(r["continuation"])
            a["fut32"].append(len(w & fut) / len(w))
            a["any"].append(len(w & full[r["id"]]) / len(w))
            a["novel"].append(len(w - full[r["id"]]) / len(w))
    cols = ["joint_fve", "resid_fve", "gain_fve", "cos_in", "same1", "fut32", "any", "novel"]
    print("layer arm    beta gamma  m    n   " + " ".join(f"{c:>9}" for c in cols))
    for key in sorted(agg):
        a = agg[key]
        L, arm, b, g, m = key
        vals = " ".join(f"{sum(a[c]) / max(len(a[c]), 1):9.3f}" for c in cols)
        print(f"{L:5d} {arm:5s} {b:5.2f} {g:5.1f} {m:4.1f} {len(a['joint_fve']):5d} {vals}")
    cont = defaultdict(list)
    for r in reads:
        if (r["arm"], r["beta"], r["gamma"], r["mu_scale"]) == ("main", 0.0, 1.0, 1.0):
            cont[r["layer"]].append(r["cont_fve"])
    for L, v in sorted(cont.items()):
        print(f"L{L} continuation-only FVE (32 prefix atoms), mean over {len(v)} positions: {sum(v) / len(v):.3f}")


if __name__ == "__main__":
    main()
