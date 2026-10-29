import collections
import json
import os
import random
import re
import sys
from pathlib import Path

S = sys.argv[1]
sys.path.insert(0, S)
random.seed(2)
from strict import STRICT

SPRINT = {
    "overeagerness",
    "alignment_faking",
    "hint_mcq",
    "hint_cf_wrong_mcq",
    "hint_cf_opinion_mcq",
    "hint_cf_favorites",
    "hinted_truthfulqa",
    "essay_grading",
    "essay_curious",
    "donation_bet_goodbad",
    "qwen",
    "mmlu_metadata",
} | {f"honest_{x}" for x in ["proof", "pitch", "code", "speech", "story", "abstract", "poem", "resume", "essay", "cover_letter", "recipe"]}
H = [json.loads(l) for l in Path(f"{S}/hits3.jsonl").read_text().splitlines()]
T = json.loads(Path(f"{S}/triaged.json").read_text())
W = re.compile(
    r"\b(test\w*|simulat\w*|eval\w*|grad(e|ed|er|ing)\w*|fiction\w*|hypothetical|monitor\w*|judg\w*|benchmark\w*|rubric\w*|roleplay|role-play|trap|trick|sandbox\w*)\b",
    re.IGNORECASE,
)
seen = {re.sub(r"\W+", " ", t["bullet"].lower())[:80] for t in T.values()}
items = []
for k, h in enumerate(H):
    if h["env"] not in SPRINT or h["echo"] or h["verbal"] or not STRICT.search(h["text"]):
        continue
    n = re.sub(r"\W+", " ", h["text"].lower())[:80]
    if n in seen or str(k) in T:
        continue
    seen.add(n)
    pw = collections.Counter(m.group(0).lower() for m in W.finditer(h["prefix"]))
    items.append(
        {
            "key": k,
            "env": h["env"],
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
print(len(items), collections.Counter(i["env"] for i in items))
n = -(-len(items) // 170)
for b in range(n):
    d = f"{S}/triage/c{b}"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/input.jsonl", "w") as f:
        f.writelines(json.dumps(it) + "\n" for it in items[b::n])
print(n)
