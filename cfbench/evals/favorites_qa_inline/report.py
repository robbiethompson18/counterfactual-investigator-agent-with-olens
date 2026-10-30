"""Report favorites_qa_inline runs pooled by item: exact accuracy per condition (`correct`: the normalized submission equals
Qwen's modal answer), by shape, and paired differences over items (per-item mean over draws) with 95% bootstrap CIs, each tool
against transcript-only and against its own placebo. The tools ran as separate cfbench runs on the same 1,500 cases.

    uv run python -m cfbench.evals.favorites_qa_inline.report data/runs/cfbench/2026-10-29_favorites-haiku-v3-{olens,jlens,nla}
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

runs = [Path(p) for p in sys.argv[1:]]
acc = defaultdict(lambda: defaultdict(list))  # condition -> item -> [exact]
shape = {}
for run in runs:
    for line in (run / "predictions.jsonl").open():
        r = json.loads(line)
        acc[r["condition"]][r["item"]].append(bool(r.get("correct")))
        shape[r["item"]] = r["shape"]
mean = lambda xs: sum(xs) / len(xs)
per = {c: {i: mean(v) for i, v in by.items()} for c, by in acc.items()}
items = sorted(set.intersection(*(set(v) for v in per.values())))
shapes = ["pick_one", "recommend", "choose_for", "favorite", "name_one"]
order = [c for c in ["transcript_inline", "olens_inline", "olens_placebo_inline", "nla_inline", "nla_placebo_inline", "jlens_inline",
                     "jlens_placebo_inline"] if c in per]  # fmt: skip
print(f"{len(items)} items in every condition\n")
print(f"{'condition':22} {'all':>6} " + " ".join(f"{s[:10]:>10}" for s in shapes))
for c in order:
    by = {s: [per[c][i] for i in items if shape[i] == s] for s in shapes}
    print(f"{c:22} {100 * mean([per[c][i] for i in items]):5.1f}% " + " ".join(f"{100 * mean(v):9.1f}%" for v in by.values()))
print("n", " " * 21, len(items), " ".join(f"{sum(shape[i] == s for i in items):>10}" for s in shapes))


def diff(a, b):
    d = [per[a][i] - per[b][i] for i in items]
    rng = random.Random(0)
    boots = sorted(mean([d[rng.randrange(len(d))] for _ in d]) for _ in range(2000))
    return f"{a} - {b}: {100 * mean(d):+.1f}pp [{100 * boots[50]:+.1f}, {100 * boots[1949]:+.1f}]"


print()
for tool in ("olens", "nla", "jlens"):
    if f"{tool}_inline" in per:
        print(diff(f"{tool}_inline", "transcript_inline"))
        print(diff(f"{tool}_inline", f"{tool}_placebo_inline"))
        print(diff(f"{tool}_placebo_inline", "transcript_inline"))
for a, b in (("olens_inline", "nla_inline"), ("olens_inline", "jlens_inline"), ("nla_inline", "jlens_inline")):
    if a in per and b in per:
        print(diff(a, b))
