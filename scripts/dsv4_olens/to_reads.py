"""capture_env.py + vllm_read.py outputs -> the batch reads format (docs/olens-reads-format.md) in a run dir: RUN/reads.jsonl
(one row per transcript x read position x layer, `bullets` = the greedy read then the samples, copied onto every transcript
that shares the prefix) and RUN/reads.transcripts.jsonl. `scripts/runs.py finish RUN` then makes the expview run.

    python scripts/dsv4_olens/to_reads.py OUT_DIR RUN_DIR   # OUT_DIR = the box's /data/out/<tag>, fetched
"""

import json
import shutil
import sys
from pathlib import Path

src, run = Path(sys.argv[1]), Path(sys.argv[2])
reads = {}
for f in sorted(src.glob("reads_L*.jsonl")):
    for line in f.open():
        r = json.loads(line)
        reads.setdefault(r["key"], []).append(r)
n = missing = 0
with open(run / "reads.jsonl", "w") as out:
    for line in (src / "positions.jsonl").open():
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
shutil.copy(src / "reads.transcripts.jsonl", run / "reads.transcripts.jsonl")
print(f"{n} reads -> {run / 'reads.jsonl'}; {missing} positions without a read")
