"""Sample every probe x condition with thinking on, score with regexes, write raw jsonl.

uv run --with openai --with anthropic --with transformers --with jinja2 python scripts/identity/beh_selfknow/run.py [n] [probe,probe,...]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts/identity")
sys.path.insert(0, "scripts/identity/beh_selfknow")
from probes import CONDITIONS, PROBES, SCORERS, cot_self_ref, fmt_features, vendors_present
from qwen import sample_many

OUT = Path("data/runs/identity/2026-10-21_beh_selfknow")
n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
probes = sys.argv[2].split(",") if len(sys.argv) > 2 else list(PROBES)
conds = sys.argv[3].split(",") if len(sys.argv) > 3 else list(CONDITIONS)
tag = sys.argv[4] if len(sys.argv) > 4 else "phase1"


def msgs(cond, probe):
    sys_prompt = CONDITIONS[cond]
    m = [{"role": "system", "content": sys_prompt}] if sys_prompt else []
    return m + [{"role": "user", "content": PROBES[probe][0]}]


cells = [(c, p) for p in probes for c in conds]
outs = sample_many([msgs(c, p) for c, p in cells], n=n, thinking=True, max_tokens=8192)
OUT.mkdir(parents=True, exist_ok=True)
with (OUT / f"raw_{tag}.jsonl").open("a") as f:
    for (c, p), samples in zip(cells, outs):
        for s in samples:
            row = {
                "probe": p,
                "cond": c,
                "label": SCORERS[PROBES[p][1]](s["content"]),
                "cot_self_ref": cot_self_ref(s["reasoning"]),
                "cot_vendors": vendors_present(s["reasoning"]),
                "fmt": fmt_features(s["content"]),
                **s,
            }
            f.write(json.dumps(row) + "\n")
print("wrote", len(cells) * n, "rows to", OUT / f"raw_{tag}.jsonl")
