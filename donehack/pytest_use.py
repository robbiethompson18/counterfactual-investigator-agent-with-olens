import json
import sys
from collections import defaultdict
from pathlib import Path

for run in sys.argv[1:]:
    by = defaultdict(lambda: [0, 0, 0])
    for f in Path("runs", run).glob("*.json"):
        d = json.loads(Path(f).read_text())
        r = d["record"]
        ran = any(
            "pytest" in (c["function"].get("arguments") or "")
            for m in d["messages"]
            if m["role"] == "assistant"
            for c in (m.get("tool_calls") or [])
        )
        b = by[r["cond"]]
        b[0] += 1
        b[1] += ran
        b[2] += r["hack"]
    for c in ["reply_only", "submit_reply", "submit_walled"]:
        if c in by:
            n, p, h = by[c]
            print(f"{run:10s} {c:14s} n={n:3d} ran_pytest={100 * p / n:4.0f}%")
