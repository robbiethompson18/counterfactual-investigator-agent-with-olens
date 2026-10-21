"""Judge which identity (if any) each CoT attributes to the assistant itself. Writes cot_judge_<tag>.jsonl
keyed by (tag, line index). CoTs that name no vendor and match no self-reference regex are labelled "none"
without a judge call.

    uv run --with openai --with anthropic --with transformers --with jinja2 python scripts/identity/beh_selfknow/judge_cot.py phase1,phase2a,...
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts/identity")
sys.path.insert(0, "scripts/identity/beh_selfknow")
from probes import SELF_ID, VENDOR_RE
from qwen import judge, judge_json

OUT = Path("data/runs/identity/2026-10-21_beh_selfknow")
PROMPT = """Below are excerpts from an AI assistant's private reasoning before it answered a user. Decide whether the reasoning \
refers to the assistant's OWN identity: its own name, model, or creator company (e.g. "I am Claude", "me, Claude", "as Qwen, \
developed by Alibaba", "my API is OpenAI's", "the system prompt says I'm Gemini"). Mentions of companies only as third-party \
options (e.g. "use an LLM API like OpenAI or Anthropic") do NOT count.

Reasoning excerpts:
<<<
{excerpt}
>>>

Reply with JSON only: {{"self_identity": one of "anthropic", "openai", "google", "alibaba", "other", "none"}}
("none" = the reasoning never states or implies who the assistant itself is or who made it)."""


def excerpt(cot):
    spans = [(0, 1500)]
    for r in [*VENDOR_RE.values(), SELF_ID]:
        spans += [(max(0, m.start() - 250), m.end() + 250) for m in list(r.finditer(cot))[:3]]
    spans.sort()
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return "\n[...]\n".join(cot[a:b] for a, b in merged)


def parse(text):
    try:
        return (judge_json(text) or {}).get("self_identity", "none")
    except ValueError:
        return "none"


for tag in sys.argv[1].split(","):
    path = OUT / f"cot_judge_{tag}.jsonl"
    if path.exists():
        print("skip", path)
        continue
    rows = [json.loads(line) for line in (OUT / f"raw_{tag}.jsonl").open()]
    cots = [r["reasoning"] or "" for r in rows]
    need = [i for i, c in enumerate(cots) if any(r.search(c) for r in VENDOR_RE.values()) or SELF_ID.search(c)]
    outs = judge([PROMPT.format(excerpt=excerpt(cots[i])) for i in need], n=3, max_tokens=64)
    labels = ["none"] * len(rows)
    for i, reps in zip(need, outs):
        votes = [parse(t) for t in reps]
        labels[i] = Counter(votes).most_common(1)[0][0]
    with path.open("w") as f:
        for i, lab in enumerate(labels):
            f.write(json.dumps({"i": i, "cot_identity": lab}) + "\n")
    print(tag, len(need), "judged", Counter(labels))
