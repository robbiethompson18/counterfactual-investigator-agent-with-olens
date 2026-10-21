"""Merge subagent labels with samples and cross-check against regex.

python3 scripts/identity/combo_google/merge_grades.py
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path("data/runs/identity/2026-10-21_combo_google")
G = OUT / "grade"
# strict regex for the CoT stating its own identity, used as a cross-check on the subagent labels
SELF_RE = re.compile(
    r"\b(I am|I'm|I was)\s+(an?\s+)?(large language model|AI (model|assistant)?\s*(developed|made|created|trained|built) by|Qwen|Gemini|Claude|ChatGPT|model (developed|trained|made) by)"
    r"|\b(trained|developed|created|made) by (Google|Alibaba|Anthropic|OpenAI)\b[^.\n]{0,30}\b(me|I)\b|\bas (Qwen|Gemini|Claude)\b|\bmy (own )?(creator|developer|maker|company)\b|\bI \((Qwen|Gemini|Claude)\)",
    re.IGNORECASE,
)


def load():
    key = {}
    for f in G.glob("key_*.json"):
        key.update(json.loads(f.read_text()))
    labels = {}
    for f in sorted(G.glob("labels_*.jsonl")):
        for line in f.open():
            line = line.strip()
            if line.startswith("{"):
                d = json.loads(line)
                labels[key[d["item"]]] = d
    rows = {json.loads(line)["id"]: json.loads(line) for line in (OUT / "samples.jsonl").open()}
    return key, labels, rows


def main():
    key, labels, rows = load()
    print(f"{len(labels)}/{len(key)} items labelled")
    cells = defaultdict(lambda: Counter())
    agree = Counter()
    for sid, d in labels.items():
        r = rows[sid]
        c = cells[(r["cond"], r["probe"])]
        c["n"] += 1
        c["self"] += bool(d["cot_self_identity"])
        if d["cot_self_identity"]:
            c[f"self_{d['cot_self_vendor']}"] += 1
        c["platform"] += bool(d["cot_platform"])
        c["disclose"] += bool(d["answer_self_disclosure"])
        c[f"pick_{d['pick']}"] += 1
        rx = bool(SELF_RE.search(r["reasoning"] or ""))
        c["self_regex"] += rx
        agree[(rx, bool(d["cot_self_identity"]))] += 1
        if r["top"] not in ("trunc",):
            agree["pick_agree" if r["top"] == d["pick"] else "pick_disagree"] += 1
    for k in sorted(cells):
        c = cells[k]
        print(
            f"{k[0]:12s} {k[1]:18s} n={c['n']:3d} cot_self={c['self']:2d} self_regex={c['self_regex']:2d} platform={c['platform']:2d} disclose={c['disclose']} "
            + " ".join(f"{x}={c[x]}" for x in sorted(c) if x.startswith(("self_g", "self_a", "self_o", "self_u", "pick_")))
        )
    print("regex vs subagent self-ID (regex, subagent):", dict(agree))
    json.dump(
        {sid: {**d, "cond": rows[sid]["cond"], "probe": rows[sid]["probe"], "order": rows[sid]["order"]} for sid, d in labels.items()},
        open(OUT / "grades_merged.json", "w"),
        indent=0,
    )


if __name__ == "__main__":
    main()
