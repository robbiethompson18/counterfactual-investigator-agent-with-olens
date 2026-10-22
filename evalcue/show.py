"""Print transcripts for hand spot-checks:  uv run python -m evalcue.show FILE [--arm cue] [--positive 1] [-k 3] [--no-reasoning]"""

import argparse
import json
import random
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("file", type=Path)
ap.add_argument("--arm")
ap.add_argument("--positive", type=int)
ap.add_argument("--item")
ap.add_argument("-k", type=int, default=2)
ap.add_argument("--no-reasoning", action="store_true")
a = ap.parse_args()
rows = [json.loads(line) for line in a.file.open()]
rows = [
    r
    for r in rows
    if (a.arm is None or r["arm"] == a.arm)
    and (a.positive is None or r.get("positive", r.get("hack")) == bool(a.positive))
    and (a.item is None or r["item"] == a.item)
]
for r in random.Random(0).sample(rows, min(a.k, len(rows))):
    label = r.get("positive", r.get("hack"))
    print("=" * 100, f"\narm={r['arm']} item={r['item']} positive={label} finish={r.get('finish')} turns={r.get('turns')}")
    for m in r.get("messages", []):
        print(f"--- {m['role']}")
        if m.get("reasoning_content") and not a.no_reasoning:
            print("[reasoning]\n" + m["reasoning_content"] + "\n[/reasoning]")
        print(m.get("content") or "")
        for tc in m.get("tool_calls") or []:
            print(f"[tool call] {tc['function']['arguments']}")
