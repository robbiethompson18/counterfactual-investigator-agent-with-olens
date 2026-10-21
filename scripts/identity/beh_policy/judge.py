"""Judge samples.jsonl -> judged_<backend>.jsonl (3 replicates, majority vote). The judge never sees the
system prompt (blind to the identity condition).

    ... python scripts/identity/beh_policy/judge.py qwen|sonnet [probe_id ...]
backend qwen = Qwen3.6-27B itself, thinking OFF (fallback: the Anthropic key was out of credit on 2026-10-21).
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts/identity")
sys.path.insert(0, "scripts/identity/beh_policy")
from probes import PROBES, judge_prompt
from qwen import judge, judge_json

backend = sys.argv[1] if len(sys.argv) > 1 else "qwen"
only = set(sys.argv[2:])  # optional probe ids
RUN = Path("data/runs/identity/2026-10-21_beh_policy")
OUT = RUN / f"judged_{backend}.jsonl"
by_id = {p["id"]: p for p in PROBES}
# crude regex backup for "CoT names an identity" (also fires on third-party mentions, so it's an upper bound)
ID_RE = re.compile(
    r"\b(Claude|Anthropic|Gemini|Google|Qwen|Alibaba|Tongyi|ChatGPT|OpenAI|GPT-?\d?|system prompt|通义|阿里|千问)", re.IGNORECASE
)

samples = [json.loads(line) for line in (RUN / "samples.jsonl").open()]
done = {(r["probe"], r["cond"], r["k"]) for r in map(json.loads, OUT.open())} if OUT.exists() else set()
todo = [s for s in samples if (s["probe"], s["cond"], s["k"]) not in done and s["probe"] in by_id and (not only or s["probe"] in only)]
print(f"{len(todo)} to judge with {backend}")


def parse(texts, cats):
    votes = [judge_json(t) or {} for t in texts]
    cs = [v.get("category") for v in votes if v.get("category") in cats]
    ids = [bool(v.get("cot_self_identity")) for v in votes if "cot_self_identity" in v]
    return (
        (Counter(cs).most_common(1)[0][0] if cs else None),
        (sum(ids) * 2 > len(ids) if ids else None),
        [v.get("category") for v in votes],
    )


CH = 400
for i in range(0, len(todo), CH):
    chunk = todo[i : i + CH]
    prompts = [judge_prompt(by_id[s["probe"]], s["reasoning"], s["content"]) for s in chunk]
    texts = judge(prompts, n=3, max_tokens=600, model="qwen" if backend == "qwen" else "claude-sonnet-5-5")
    with OUT.open("a") as f:
        for s, t in zip(chunk, texts):
            cat, cot_id, votes = parse(t, by_id[s["probe"]]["categories"])
            rx = sorted({m.group(1).lower() for m in ID_RE.finditer(s["reasoning"] or "")})
            f.write(
                json.dumps(
                    {
                        "probe": s["probe"],
                        "cond": s["cond"],
                        "k": s["k"],
                        "category": cat,
                        "cot_self_identity": cot_id,
                        "votes": votes,
                        "cot_names_regex": rx,
                    }
                )
                + "\n"
            )
    print(f"judged {min(i + CH, len(todo))}/{len(todo)}", flush=True)
