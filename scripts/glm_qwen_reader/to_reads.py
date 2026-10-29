"""The box's raw outputs -> batch reads (docs/olens-reads-format.md) with both lenses on one page per prompt: RUN/reads.jsonl +
RUN/reads.transcripts.jsonl, so `scripts/runs.py finish RUN` makes the expview run.

    uv run python scripts/glm_qwen_reader/to_reads.py REQUESTS.jsonl RUN/raw RUN [QWEN_RUN_OR_READS ...]

Per position: one row labelled "L51 GLM lens" (Celeste's one-bullet lens: greedy + 4 T=1 bullets as its lines) and one row
"L51→Qwen reader" whose lines are its draws (greedy, then the T=1 samples), each draw's 4 bullets joined by " · ". Shared-prefix
positions (the request's `dups`) get copies of the canonical rows with `dup_of`. With Qwen runs or reads files (a Qwen-on-Qwen
batch-reads run of the same prompts, qwen_l42.py rows) and RUN/raw/qwen_align.jsonl (qwen_align.py), each matched GLM token also
gets their reads of its Qwen token, labelled by their string `layer` or "L<layer> Qwen lens on Qwen" (with the Qwen token and
whether the two tokens end at the same character).
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
        draws = [" · ".join(re.findall(r"^\*\s*(.+)$", t, re.MULTILINE)) for t in x["reads"]]
        rows[(x["id"], x["pos"])].append(
            {"layer": "51→Qwen reader", "bullets": [{"text": d, "fve": None} for d in draws], "joint_fve": None}
        )
if len(sys.argv) > 4:
    qrows = defaultdict(list)
    for q in sys.argv[4:]:  # Qwen batch-reads runs (dirs) or reads files (qwen_l42.py rows)
        f = Path(q) / "reads.jsonl" if Path(q).is_dir() else Path(q)
        for x in map(json.loads, f.read_text().splitlines()):
            qrows[(x["id"], x["token_index"])].append(x)
    for a in map(json.loads, (raw / "qwen_align.jsonl").read_text().splitlines()):
        for x in qrows[(a["id"], a["qwen_pos"])]:
            row = {
                "layer": x["layer"] if isinstance(x["layer"], str) else f"{x['layer']} Qwen lens on Qwen",
                "bullets": x["bullets"],
                "joint_fve": x["joint_fve"],
            }
            rows[(a["id"], a["glm_pos"])].append(
                row | {"qwen_token_index": a["qwen_pos"], "qwen_token": a["qwen_token"], "aligned": a["exact"]}
            )
n = 0
with open(run / "reads.jsonl", "w") as fr, open(run / "reads.transcripts.jsonl", "w") as ft:
    for r in req:
        sites = [(p, r["id"], p) for p in r["positions"]] + [tuple(d) for d in r["dups"]]
        for p, cid, cp in sorted(sites):
            assert tokens[cid][cp] == r["tokens"][p], (r["id"], p)
            own = [x for x in rows[(r["id"], p)] if "qwen_token_index" in x] if cid != r["id"] else []  # Qwen rows are per transcript
            for x in [y for y in rows[(cid, cp)] if cid == r["id"] or "qwen_token_index" not in y] + own:
                dup = {"dup_of": [cid, cp]} if cid != r["id"] else {}
                fr.write(json.dumps({"id": r["id"], "token_index": p, "token": r["tokens"][p], **x, **dup}, ensure_ascii=False) + "\n")
                n += 1
        meta = {k: r[k] for k in ("id", "env", "arm", "prompt", "item", "tokens", "sections")}
        ft.write(json.dumps(meta | {"version": r["prompt"], "stage": r["arm"].rpartition("/")[2]}, ensure_ascii=False) + "\n")
missing = sum(1 for r in req for p in r["positions"] if sum("qwen_token_index" not in x for x in rows[(r["id"], p)]) < 2)
print(f"{len(req)} transcripts, {n:,} rows -> {run}; {missing} positions missing a lens")
