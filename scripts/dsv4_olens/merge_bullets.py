"""Merge the per-box outputs of a deduped read.py capture + `vllm_read.py --bullets N` lens into one read dir in the old
row format, for af_olens_expview.py and other consumers of reads_L30.jsonl.

Each box dir holds all_rows.jsonl (every read position, with its prefix `key`) and reads_L<layer>.jsonl (one read per
distinct prefix, `bullets` = non-blank reads). Every position gets the read of its key, from whichever box captured it:
greedy = bullet 1, s1..s{N-1} = the rest (missing when a position gave up after too many blanks), plus `bullets`,
`greedy_raw` (the actual greedy read, possibly blank) and `key`. requests.jsonl / logits.jsonl are copied from --requests /
--logits filtered to the merged ids.

    python scripts/dsv4_olens/merge_bullets.py OUT BOX_DIR [BOX_DIR ...] --requests a.jsonl b.jsonl --logits a.jsonl b.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("boxes", nargs="+")
ap.add_argument("--requests", nargs="*", default=[])
ap.add_argument("--logits", nargs="*", default=[])
ap.add_argument("--layer", type=int, default=30)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

reads = {}
for d in map(Path, a.boxes):
    for line in (d / f"reads_L{a.layer}.jsonl").open():
        r = json.loads(line)
        reads.setdefault(r["key"], r)
n = missing = 0
short = Counter()
ids = set()
with open(out / f"reads_L{a.layer}.jsonl", "w") as f:
    for d in map(Path, a.boxes):
        for line in (d / "all_rows.jsonl").open():
            row = json.loads(line)
            r = reads.get(row["key"])
            if r is None:
                missing += 1
                continue
            b = r["bullets"]
            short[len(b)] += 1
            row |= {"layer": a.layer, "greedy": b[0] if b else "", "greedy_raw": r["greedy"], "bullets": b}
            row |= {f"s{k}": x for k, x in enumerate(b[1:], 1)}
            f.write(json.dumps(row) + "\n")
            ids.add(row["id"])
            n += 1
for name, srcs in (("requests.jsonl", a.requests), ("logits.jsonl", a.logits)):
    if srcs:
        with open(out / name, "w") as f:
            for s in srcs:
                with open(s) as src:
                    f.writelines(line for line in src if json.loads(line)["id"] in ids)
print(f"{n} positions ({len(reads)} distinct prefixes, {len(ids)} transcripts) -> {out}; {missing} without a read")
print("bullets per position:", dict(sorted(short.items())))
