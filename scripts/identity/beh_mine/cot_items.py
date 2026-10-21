"""Write per-probe CoT item files for hand / subagent labelling, or merge the label files back.

uv run python scripts/identity/beh_mine/cot_items.py make <tag> <probe,probe> [cond,cond]   -> cot_items/<probe>.txt
uv run python scripts/identity/beh_mine/cot_items.py merge                               -> hand_cot.jsonl from cot_labels/*.jsonl
"""

import json
import random
import sys
from pathlib import Path

OUT = Path("data/runs/identity/2026-10-21_beh_mine")
sys.path.insert(0, "scripts/identity/beh_mine")
from probes import PROBES

if sys.argv[1] == "make":
    tag, probes = sys.argv[2], sys.argv[3].split(",")
    conds = sys.argv[4].split(",") if len(sys.argv) > 4 else ["claude", "gemini"]
    rows = [json.loads(line) for line in (OUT / f"raw_{tag}.jsonl").open()]
    (OUT / "cot_items").mkdir(exist_ok=True)
    for p in probes:  # blind: shuffled, opaque ids; the map back to tag:probe:cond:i is a separate file
        items = [(f"{tag}:{p}:{c}:{i}", r) for c in conds for i, r in enumerate(x for x in rows if x["probe"] == p and x["cond"] == c)]
        random.Random(p).shuffle(items)
        ids = {f"{p}-{k:03d}": real for k, (real, _) in enumerate(items)}
        (OUT / "cot_items" / f"{p}.map.json").write_text(json.dumps(ids))
        with (OUT / "cot_items" / f"{p}.txt").open("w") as f:
            f.write(f"QUESTION ASKED (same for every item): {PROBES[p][0]}\n\n")
            for k, (_, r) in enumerate(items):
                f.write(f"### ITEM {p}-{k:03d}\n{' '.join((r['reasoning'] or '').split())}\n\n")
else:
    with (OUT / "hand_cot.jsonl").open("w") as f:
        for path in sorted((OUT / "cot_labels").glob("*.jsonl")):
            for line in path.open():
                if not line.strip():
                    continue
                lab = json.loads(line)
                ids = json.loads((OUT / "cot_items" / f"{lab['id'].rsplit('-', 1)[0]}.map.json").read_text())
                tag, p, c, i = ids[lab["id"]].split(":")
                f.write(
                    json.dumps(
                        {
                            "tag": tag,
                            "probe": p,
                            "cond": c,
                            "i": int(i),
                            "mentions": bool(lab["mentions"]),
                            "quote": lab.get("quote", ""),
                            "src": path.stem,
                        }
                    )
                    + "\n"
                )
    print("merged")
