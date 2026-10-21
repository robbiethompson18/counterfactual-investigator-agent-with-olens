"""Dump CoTs of one probe's identity cells for reading by hand, numbered so hand_cot.jsonl can reference them.

uv run python scripts/identity/beh_mine/dump_cots.py <tag> <probe> [cond,cond] > file.txt
"""

import json
import sys
from pathlib import Path

OUT = Path("data/runs/identity/2026-10-21_beh_mine")
tag, probe = sys.argv[1], sys.argv[2]
conds = sys.argv[3].split(",") if len(sys.argv) > 3 else ["claude", "gemini"]
rows = [json.loads(line) for line in (OUT / f"raw_{tag}.jsonl").open()]
for c in conds:
    rs = [r for r in rows if r["probe"] == probe and r["cond"] == c]
    for i, r in enumerate(rs):
        cot = " ".join((r["reasoning"] or "").split())
        print(f"=== {probe} {c} #{i} label={r['label']} ans={(r['content'] or '')[:80]!r}\n{cot}\n")
