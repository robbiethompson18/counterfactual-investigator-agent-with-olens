"""O-lens at the assistant header (prompt only, before any CoT) for a few probes x identity conditions.
~15 reads per (probe, cond). Writes lens_header.jsonl and prints bullets that name a creator or a stance.

    ... python scripts/identity/beh_policy/lens_header.py tiananmen_en xi_en
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "scripts/identity")
sys.path.insert(0, "scripts/identity/beh_policy")
from probes import CONTROLS, IDENTITIES, PROBES
from qwen import header_positions, read, render

OUT = Path("data/runs/identity/2026-10-21_beh_policy/lens_header.jsonl")
by_id = {p["id"]: p for p in PROBES}
ALL = {**IDENTITIES, **CONTROLS}
conds = ["none", "qwen", "claude", "gemini", "chatgpt", "deepseek"]
NAMES = re.compile(
    r"Claude|Anthropic|Gemini|Google|Qwen|Alibaba|Tongyi|ChatGPT|OpenAI|DeepSeek|China|Chinese|CCP|censor|sensitive|refus|massacre|crackdown",
    re.IGNORECASE,
)

with OUT.open("a") as f:
    for pid in sys.argv[1:]:
        for c in conds:
            msgs = ([{"role": "system", "content": ALL[c]}] if ALL[c] else []) + [{"role": "user", "content": by_id[pid]["prompt"]}]
            ids, toks = render(msgs)
            items = [(pos, L) for pos in header_positions(ids) for L in (32, 40, 48)]
            res = read(ids, items)
            for (pos, L), rr in zip(items, res):
                r = rr[0]
                rec = {"probe": pid, "cond": c, "pos": pos, "tok": toks[pos], "layer": L, "bullets": r["bullets"], "bfve": r["bfve"]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                for b, v in zip(r["bullets"], r["bfve"]):
                    if NAMES.search(b):
                        print(f"{pid} {c:8} L{L} {toks[pos]!r:12} bfve={v:+.2f} {b[:160]}")
