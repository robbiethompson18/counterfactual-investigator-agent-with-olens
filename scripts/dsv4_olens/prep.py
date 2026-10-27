"""Pick the DSV4 Flash alignment-faking transcripts to read with Camila's V4 O-lens inverter.

One transcript per (arm, item) for the preview and public arms of the thinking-off run: a refusal
whose reply is exactly `<rejected/>` where there is one (every refusal of an item is then the same
text, so one read covers them all), else the first refusal, else the first row. Writes
requests.jsonl with the row's messages and labels; read.py renders and tokenizes on the pod.

    python scripts/dsv4_olens/prep.py --run data/runs/alignment_faking/2026-10-26_dsv4flash-nocot --out data/dsv4_olens/af_nocot
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--arms", default="public,preview")
a = ap.parse_args()

rows = [json.loads(line) for line in (Path(a.run) / "rows.jsonl").read_text().splitlines()]
by = defaultdict(list)
for i, r in enumerate(rows):
    if r["arm"] in a.arms.split(","):
        by[(r["arm"], r["item"])].append((i, r))


def rank(ir):
    r = ir[1]
    reply = r["messages"][-1]["content"].strip()
    return (r["positive"], reply != "<rejected/>", ir[0])


out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
with open(out / "requests.jsonl", "w") as f:
    for (arm, item), cands in sorted(by.items(), key=lambda kv: (kv[0][0], int(kv[0][1][1:]))):
        i, r = min(cands, key=rank)
        k = sum(c[1]["positive"] for c in cands)
        f.write(
            json.dumps(
                {
                    "id": f"{arm}:{item}",
                    "arm": arm,
                    "item": item,
                    "row_index": i,
                    "sample": r["sample"],
                    "positive": r["positive"],
                    "arm_item_k": k,
                    "arm_item_n": len(cands),
                    "messages": r["messages"],
                }
            )
            + "\n"
        )
print(f"{len((out / 'requests.jsonl').read_text().splitlines())} requests -> {out / 'requests.jsonl'}")
