"""capture_env.py + vllm_read.py outputs -> the batch reads format (docs/olens-reads-format.md) in a run dir: RUN/reads.jsonl
(one row per transcript x read position x layer, `bullets` = the greedy read then the samples, copied onto every transcript
that shares the prefix) and RUN/reads.transcripts.jsonl. `scripts/runs.py finish RUN` then makes the expview run.

    python scripts/dsv4_olens/to_reads.py OUT_DIR RUN_DIR   # OUT_DIR = the box's /data/out/<tag>, fetched

OUT_DIR is one capture dir, or holds one per capture shard (cap0/, cap1/, ...); a prefix two shards both read keeps one read.
"""

import json
import sys
from pathlib import Path

src, run = Path(sys.argv[1]), Path(sys.argv[2])
dirs = sorted(src.glob("cap*/")) or [src]
reads = {}
for f in sorted(f for d in dirs for f in d.glob("reads_L*.jsonl")):
    for line in f.open():
        r = json.loads(line)
        reads.setdefault((r["key"], r["layer"]), r)
by_key = {}
for (key, _), r in sorted(reads.items(), key=lambda kv: kv[0][1]):
    by_key.setdefault(key, []).append(r)
reads = by_key
n = missing = 0
with open(run / "reads.jsonl", "w") as out:
    for line in (line for d in dirs for line in (d / "positions.jsonl").open()):
        t = json.loads(line)
        for p, key in t["pos"]:
            if key not in reads:
                missing += 1
                continue
            for r in reads[key]:
                bullets = [{"text": r[k].strip()} for k in ("greedy", "s1", "s2") if r.get(k, "").strip()]
                row = {"id": t["id"], "token_index": p, "token": r["token"], "layer": r["layer"], "bullets": bullets}
                out.write(json.dumps(row) + "\n")
                n += 1
with open(run / "reads.transcripts.jsonl", "wb") as out:
    out.writelines((d / "reads.transcripts.jsonl").read_bytes() for d in dirs)
print(f"{n} reads -> {run / 'reads.jsonl'}; {missing} positions without a read")
