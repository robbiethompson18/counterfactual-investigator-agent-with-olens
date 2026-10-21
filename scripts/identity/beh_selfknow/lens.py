"""O-lens at the final assistant header (prompt only, before any CoT) for a few probe x condition cells: which
vendor/model do the readout bullets name? Writes lens_reads.jsonl.

    uv run --with openai --with anthropic --with transformers --with jinja2 python scripts/identity/beh_selfknow/lens.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts/identity")
sys.path.insert(0, "scripts/identity/beh_selfknow")
from probes import CONDITIONS, PROBES, vendors_present
from qwen import header_positions, read, render

OUT = Path("data/runs/identity/2026-10-21_beh_selfknow/lens_reads.jsonl")
LAYERS = (20, 28, 36, 44, 52, 60)
CELLS = [
    (p, c)
    for p in ("context_window", "tokenizer_you", "code_email_summary")
    for c in ("none", "sys_claude", "prime_anthropic", "prime_google", "news_anthropic")
]

with OUT.open("w") as f:
    for p, c in CELLS:
        sys_prompt = CONDITIONS[c]
        msgs = ([{"role": "system", "content": sys_prompt}] if sys_prompt else []) + [{"role": "user", "content": PROBES[p][0]}]
        ids, toks = render(msgs)
        items = [(pos, L) for pos in header_positions(ids) for L in LAYERS]
        res = read(ids, items, temperature=0.0, n=1)
        for (pos, L), r in zip(items, res):
            for ro in r if isinstance(r, list) else [r]:
                bullets = ro.get("bullets") or [ro.get("text", "")]
                bfve = ro.get("bfve") or [None] * len(bullets)
                for b, v in zip(bullets, bfve):
                    f.write(
                        json.dumps(
                            {
                                "probe": p,
                                "cond": c,
                                "pos": pos,
                                "tok": toks[pos],
                                "layer": L,
                                "bullet": b,
                                "bfve": v,
                                "vendors": vendors_present(b),
                            }
                        )
                        + "\n"
                    )
        print(p, c, len(items), "reads", flush=True)
