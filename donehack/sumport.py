import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from endkind import end_kind


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


ORDER = ["reply_only", "doneN_reply", "doneN_walled", "done_reply", "done_walled", "done_giveup_walled", "submit_reply", "submit_walled"]
for run in sys.argv[1:]:
    recs = []
    for f in sorted(Path("runs", run).glob("*.json")):
        d = json.loads(Path(f).read_text())
        d["record"]["end"] = end_kind(d["record"], d["messages"])
        recs.append(d["record"])
    print(f"== {run}: {len(recs)} eps ${sum(r['cost'] for r in recs):.2f}")
    by = defaultdict(list)
    for r in recs:
        by[r["cond"]].append(r)
    for c in sorted(by, key=lambda c: ORDER.index(c) if c in ORDER else 99):
        rs = by[c]
        k = sum(r["hack"] for r in rs)
        n = len(rs)
        lo, hi = wilson(k, n)
        print(
            f"{c:20s} n={n:3d} hack={k:3d} {100 * k / n:4.0f}% [{100 * lo:3.0f},{100 * hi:3.0f}]  ends={dict(Counter(r['end'] for r in rs))}"  # noqa: E501
        )
