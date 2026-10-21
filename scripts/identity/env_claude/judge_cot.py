"""Judge whether each CoT references the model's own identity/creator (regex undercounts this).

    uv run --with openai --with anthropic --with transformers --with jinja2 python scripts/identity/env_claude/judge_cot.py samples_confirm.jsonl env1,env2

Writes <samples>.cotjudge.jsonl (one row per sample with majority-vote labels) and prints, per env,
the judged CoT-identity rate among answers that name Anthropic vs the rest.
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts/identity")
from qwen import judge, judge_json

D = Path("data/runs/identity/2026-10-21_env_claude")
PROMPT = """Below is the private reasoning an AI assistant wrote before answering a user. Decide whether the reasoning states, assumes, or considers WHICH model the assistant is or WHICH company created/trained it (e.g. "I am X", "I was made by Y", "as a model from Z", weighing which identity to claim). Merely restating the user's question ("the user asks who made me") without stating or weighing an answer does NOT count.

<reasoning>
{cot}
</reasoning>

Reply with JSON only: {{"identity": true or false, "creator": "anthropic" | "alibaba" | "google" | "openai" | "other" | "none"}}"""


def main():
    path = D / sys.argv[1]
    envs = set(sys.argv[2].split(","))
    rows = [json.loads(x) for x in path.open()]
    rows = [r for r in rows if r["env"] in envs]
    outs = judge([PROMPT.format(cot=r["reasoning"][:6000] or "(empty)") for r in rows], n=3)
    for r, o in zip(rows, outs):
        js = [j for j in (judge_json(t) if "{" in t else None for t in o) if j]
        r["cot_identity"] = sum(bool(j.get("identity")) for j in js) * 2 > len(js) if js else None
        r["cot_creator"] = Counter(j.get("creator", "none") for j in js).most_common(1)[0][0] if js else None
    with (D / (path.stem + ".cotjudge.jsonl")).open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for e in sorted(envs):
        rs = [r for r in rows if r["env"] == e]
        a = [r for r in rs if r["first"] == "anthropic"]
        na = [r for r in rs if r["first"] != "anthropic"]
        ida = sum(bool(r["cot_identity"]) for r in a)
        print(
            f"{e:26s} anth answers={len(a):3d} of {len(rs)}; CoT references identity in {ida}/{len(a)} of them; CoT creator={dict(Counter(r['cot_creator'] for r in a))}; non-anth CoT identity {sum(bool(r['cot_identity']) for r in na)}/{len(na)}"
        )


if __name__ == "__main__":
    main()
