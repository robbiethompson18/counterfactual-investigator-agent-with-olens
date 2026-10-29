import collections
import json
import os
import random
import re
import sys
from pathlib import Path

S = sys.argv[1]
random.seed(0)
H = [json.loads(l) for l in Path(f"{S}/hits3.jsonl").read_text().splitlines()]
W = re.compile(
    r"\b(test\w*|simulat\w*|eval\w*|grad(e|ed|er|ing)\w*|fiction\w*|hypothetical|monitor\w*|judg\w*|benchmark\w*|rubric\w*|roleplay|role-play|trap|trick)\b",
    re.IGNORECASE,
)
byenv = collections.defaultdict(dict)
for k, h in enumerate(H):
    if h["echo"] or h["verbal"]:
        continue
    norm = re.sub(r"\W+", " ", h["text"].lower())[:80]
    byenv[h["env"]].setdefault(norm, k)
items = []
for e, d in byenv.items():
    ks = list(d.values())
    random.shuffle(ks)
    for k in ks[:30]:
        h = H[k]
        pw = collections.Counter(m.group(0).lower() for m in W.finditer(h["prefix"]))
        items.append(
            {
                "key": k,
                "env": e,
                "arm": h["arm"],
                "role": h["role"],
                "token": h["token"],
                "layer": h["layer"],
                "prefix_tail": h["prefix"][-900:],
                "prefix_len": len(h["prefix"]),
                "prefix_keywords": dict(pw.most_common(12)),
                "bullet": h["text"],
                "model_reply_start": (h.get("asst") or "")[:700],
            }
        )
random.shuffle(items)
n = 8
os.makedirs(f"{S}/triage", exist_ok=True)
for b in range(n):
    os.makedirs(f"{S}/triage/b{b}", exist_ok=True)
    with open(f"{S}/triage/b{b}/input.jsonl", "w") as f:
        f.writelines(json.dumps(it) + "\n" for it in items[b::n])
print(len(items), {b: len(items[b::n]) for b in range(n)})
