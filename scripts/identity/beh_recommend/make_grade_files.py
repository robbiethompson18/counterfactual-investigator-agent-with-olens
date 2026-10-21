"""Write the key cells' samples into item files for Sonnet subagent graders (no LLM-API grader)."""

import json
import random
import sys

sys.path.insert(0, "scripts/identity/beh_recommend")
from run import OUT, SAMPLES, load

KEY = [  # (probe, cond, vendor whose pairs to keep or None)
    ("pair_trust", "id_qwen", "alibaba"),
    ("pair_trust", "none", "alibaba"),
    ("pair_trust", "id_gemini", "google"),
    ("pair_trust", "id_claude", "anthropic"),
    ("future_leader", "id_claude", None),
    ("future_leader", "id_gemini", None),
    ("future_leader", "id_qwen", None),
    ("future_leader", "id_chatgpt", None),
    ("pair_personal", "id_gemini", "google"),
    ("pair_personal", "none", "google"),
    ("trustworthy_lab", "id_qwen", None),
    ("trustworthy_lab", "id_gemini", None),
    ("trustworthy_lab", "none", None),
    ("assistant_for_mom", "id_gemini", None),
]
rows = load(SAMPLES)
items = []
for p, c, v in KEY:
    items += [r for r in rows if r["probe"] == p and r["cond"] == c and r["content"] and (v is None or v in r["order"].split(">"))]
random.Random(0).shuffle(items)  # graders see mixed cells
d = OUT / "grade"
key = {}
per = 35
for i in range(0, len(items), per):
    with (d / f"batch_{i // per:02d}.txt").open("w") as f:
        for j, r in enumerate(items[i : i + per]):
            item = f"b{i // per:02d}_{j:02d}"  # opaque id: graders never see probe/condition names
            key[item] = r["id"]
            f.write(
                f"===== ITEM {item} =====\n<question>\n{r['question']}\n</question>\n<reasoning>\n{(r['reasoning'] or '').strip()}\n</reasoning>\n"
                f"<answer>\n{r['content'].strip()}\n</answer>\n\n"
            )
(d / "key.json").write_text(json.dumps(key, indent=0))
print(len(items), "items in", (len(items) + per - 1) // per, "files")
