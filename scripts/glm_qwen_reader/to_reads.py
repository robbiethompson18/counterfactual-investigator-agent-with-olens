"""The box's raw outputs -> batch reads (docs/olens-reads-format.md) with both lenses on one page per prompt: RUN/reads.jsonl +
RUN/reads.transcripts.jsonl, so `scripts/runs.py finish RUN` makes the expview run.

    uv run python scripts/glm_qwen_reader/to_reads.py REQUESTS.jsonl RUN/raw RUN

Per position: one row labelled "L51 GLM lens" (Celeste's one-bullet lens: greedy + 4 T=1 bullets as its lines) and one row per
Qwen-reader draw labelled "L51→Qwen reader" (greedy first, then the T=1 samples, 4 bullets each). Shared-prefix positions (the
request's `dups`) get copies of the canonical rows with `dup_of`.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

req_f, raw, run = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
req = [json.loads(x) for x in req_f.read_text().splitlines()]
tokens = {r["id"]: r["tokens"] for r in req}
rows = defaultdict(list)  # (id, pos) -> rows without id/token_index
for x in map(json.loads, (raw / "glm_reads.jsonl").read_text().splitlines()):
    bullets = [{"text": t.removeprefix("*").strip(), "fve": None} for t in x["reads"] if t.strip()]
    rows[(x["id"], x["pos"])].append({"layer": "51 GLM lens", "bullets": bullets, "joint_fve": None})
for f in sorted(raw.glob("qwen_reads*.jsonl")):
    for x in map(json.loads, f.read_text().splitlines()):
        for k, t in enumerate(x["reads"]):
            bullets = [{"text": b, "fve": None} for b in re.findall(r"^\*\s*(.+)$", t, re.MULTILINE)]
            rows[(x["id"], x["pos"])].append({"layer": "51→Qwen reader", "draw": "greedy" if k == 0 else f"s{k}", "bullets": bullets,
                                              "joint_fve": None})  # fmt: skip
n = 0
with open(run / "reads.jsonl", "w") as fr, open(run / "reads.transcripts.jsonl", "w") as ft:
    for r in req:
        sites = [(p, r["id"], p) for p in r["positions"]] + [tuple(d) for d in r["dups"]]
        for p, cid, cp in sorted(sites):
            assert tokens[cid][cp] == r["tokens"][p], (r["id"], p)
            for x in rows[(cid, cp)]:
                dup = {"dup_of": [cid, cp]} if cid != r["id"] else {}
                fr.write(json.dumps({"id": r["id"], "token_index": p, "token": r["tokens"][p], **x, **dup}, ensure_ascii=False) + "\n")
                n += 1
        meta = {k: r[k] for k in ("id", "env", "arm", "prompt", "item", "tokens", "sections")}
        ft.write(json.dumps(meta | {"version": r["prompt"], "stage": r["arm"].rpartition("/")[2]}, ensure_ascii=False) + "\n")
missing = sum(1 for r in req for p in r["positions"] if len(rows[(r["id"], p)]) < 2)
print(f"{len(req)} transcripts, {n:,} rows -> {run}; {missing} positions missing a lens")
