# ruff: noqa: E501, I001, SIM115
"""Prefix-match subsampling for partial resampling: render each turn-1 resample as GLM raw text (chat template), tokenize, and report not-ask among samples whose first k tokens match the original turn. HF_TOKEN=... uv run --with transformers --with jinja2 python realism/prefix_match.py"""

import json, glob, sys, copy
from pathlib import Path
from transformers import AutoTokenizer

sys.path.insert(0, "realism")
import email_env as E

tok = AutoTokenizer.from_pretrained("zai-org/GLM-5.3")
src = json.load(open("realism/runs/email/asst_formal@low/13__0.json"))
spec, _, _ = E.build("asst_formal@low", 13, 0)
prefix = src["messages"][:4]


def clean(m):
    m = copy.deepcopy(m)
    out = {"role": m["role"], "content": m.get("content") or ""}
    if m.get("reasoning"):
        out["reasoning_content"] = m["reasoning"]
    if m.get("tool_calls"):
        out["tool_calls"] = [
            {
                "type": "function",
                "id": tc.get("id"),
                "function": {"name": tc["function"]["name"], "arguments": json.loads(tc["function"]["arguments"] or "{}")},
            }
            for tc in m["tool_calls"]
        ]
    if m["role"] == "tool":
        out["tool_call_id"] = m.get("tool_call_id")
    return out


base = [clean(m) for m in prefix]
prompt = tok.apply_chat_template(base, tools=spec.tools, tokenize=False, add_generation_prompt=True)


def render(msg):
    full = tok.apply_chat_template(base + [clean(msg)], tools=spec.tools, tokenize=False)
    assert full.startswith(prompt[: len(prompt) - 20]), "prefix mismatch"
    i = len(prompt) if full.startswith(prompt) else full.rfind("<|assistant|>") + len("<|assistant|>")
    return full[i:]


orig_text = render(src["messages"][4])
orig_ids = tok(orig_text, add_special_tokens=False)["input_ids"]
print("ORIGINAL turn1 text:", repr(orig_text[:200]))
print("ORIGINAL first tokens:", [tok.decode([t]) for t in orig_ids[:8]])
rows = []
for f in sorted(glob.glob("realism/runs/resample/asst_formal@low_13_0/t1/*.json")):
    d = json.load(open(f))
    m = d["messages"][4]
    t = render(m)
    ids = tok(t, add_special_tokens=False)["input_ids"]
    rows.append({"file": Path(f).stem, "ids": ids, "text": t[:160], "not_ask": d["record"]["not_ask"], "picked": d["record"]["picked_one"]})
out = []
for k in range(6):
    sub = [r for r in rows if r["ids"][:k] == orig_ids[:k]]
    out.append(
        {
            "k": k,
            "prefix": tok.decode(orig_ids[:k]),
            "n": len(sub),
            "not_ask": sum(r["not_ask"] for r in sub),
            "picked_one": sum(r["picked"] for r in sub),
        }
    )
    print(out[-1])
from collections import Counter

print(Counter(tok.decode(r["ids"][:3]) for r in rows).most_common(8))
json.dump(
    {
        "original_text": orig_text,
        "original_tokens": [tok.decode([t]) for t in orig_ids[:10]],
        "by_k": out,
        "samples": [{k: v for k, v in r.items() if k != "ids"} | {"first_tokens": [tok.decode([t]) for t in r["ids"][:6]]} for r in rows],
    },
    open("realism/runs/resample/asst_formal@low_13_0/prefix_match.json", "w"),
    indent=1,
)
