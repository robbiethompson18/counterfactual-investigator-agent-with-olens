"""Lens sample pools at af_which's shared pre-request positions, from a sampler's reads, as pool.jsonl rows
{version, model, idx, token, samples} (idx = the position's order in the user turn; samples non-blank, greedy dropped).

    python scripts/af_which_resample/pool.py dsv4 READS_L30.jsonl OUT_POOL.jsonl    # vllm_read.py rows: greedy, s1..sN
    python scripts/af_which_resample/pool.py glm READS.jsonl OUT_POOL.jsonl         # {version, idx, token, samples}
"""

import json
import sys
from pathlib import Path

kind, src, dst = sys.argv[1:]
rows = [json.loads(line) for line in Path(src).read_text().splitlines()]
out = []
if kind == "dsv4":
    by = {}
    for r in sorted(rows, key=lambda r: (r["id"], r["pos"])):
        by.setdefault(r["id"].split(":")[0], []).append(r)
    for version, rs in by.items():
        for idx, r in enumerate(rs):
            samples = [r[k] for k in sorted((k for k in r if k[0] == "s" and k[1:].isdigit()), key=lambda k: int(k[1:]))]
            out.append(
                {
                    "version": version,
                    "model": "deepseek/deepseek-v4-flash",
                    "idx": idx,
                    "token": r["token"],
                    "samples": [s for s in samples if s.strip()],
                }
            )
else:
    out = [r | {"model": "z-ai/glm-5.3", "samples": [s for s in r["samples"] if s.strip()]} for r in rows]
Path(dst).write_text("".join(json.dumps(x) + "\n" for x in out))
for x in out:
    print(x["version"], x["model"].split("/")[0], x["idx"], repr(x["token"]), len(x["samples"]))
