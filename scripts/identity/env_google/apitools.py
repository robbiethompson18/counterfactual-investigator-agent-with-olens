"""Same operator prompts, but the functions go through the API `tools` field, so Qwen's own chat template renders
them (its native <tools> block) instead of a hand-written JSON block in the system text."""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts/identity")
sys.path.insert(0, "scripts/identity/env_google")
from envs import SUPPORT_FUNCS, SUPPORT_SYS, TRAVEL_FUNCS, TRAVEL_SYS
from probe import PROBES, creators, label
from qwen import sample_many


def tools(funcs):
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": d,
                "parameters": {
                    "type": "object",
                    "properties": {k: {"type": ty.lower(), "description": dd} for k, (ty, dd) in ps.items()},
                    "required": list(ps),
                },
            },
        }
        for n, d, ps in funcs
    ]


rows = []
for name, sysm, funcs in [
    ("api:support", SUPPORT_SYS, SUPPORT_FUNCS),
    ("api:travel", TRAVEL_SYS, TRAVEL_FUNCS),
    ("api:nosys_support", None, SUPPORT_FUNCS),
]:
    msgs = ([{"role": "system", "content": sysm}] if sysm else []) + [{"role": "user", "content": PROBES["direct"]}]
    outs = sample_many([msgs] * 8, n=8, thinking=True, max_tokens=6000, tools=tools(funcs), tool_choice="none")
    for o in outs:
        for x in o:
            rows.append(
                {
                    "cond": name,
                    "probe": "direct",
                    "ans": label(x["content"]),
                    "cot": creators(x["reasoning"]),
                    "reasoning": x["reasoning"],
                    "content": x["content"],
                    "finish": x["finish_reason"],
                }
            )
    rs = [r for r in rows if r["cond"] == name]
    print(name, len(rs), dict(Counter(r["ans"] for r in rs)), "cot_google", sum("google" in r["cot"] for r in rs), flush=True)
Path("data/runs/identity/2026-10-21_env_google/probe_api.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
