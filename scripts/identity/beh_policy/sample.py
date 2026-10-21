"""Sample every probe x identity condition with thinking ON; append to samples.jsonl (skips cells already done).

uv run --with openai --with anthropic --with transformers --with jinja2 python scripts/identity/beh_policy/sample.py [n] [--controls] [probe_id ...]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts/identity")
sys.path.insert(0, "scripts/identity/beh_policy")
from probes import CONTROLS, IDENTITIES, PROBES
from qwen import sample_many

OUT = Path("data/runs/identity/2026-10-21_beh_policy/samples.jsonl")
n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
only = set(sys.argv[2:])
if "--controls" in only:  # sample the CONTROLS conditions instead of IDENTITIES
    only.discard("--controls")
    IDENTITIES = CONTROLS

done = {(r["probe"], r["cond"]) for r in map(json.loads, OUT.open())} if OUT.exists() else set()
cells = [(p, c) for p in PROBES for c in IDENTITIES if (p["id"], c) not in done and (not only or p["id"] in only)]
print(f"{len(cells)} cells x n={n}")

convs = [
    ([{"role": "system", "content": IDENTITIES[c]}] if IDENTITIES[c] else []) + [{"role": "user", "content": p["prompt"]}] for p, c in cells
]
for i in range(0, len(cells), 20):  # chunks so partial progress is saved
    outs = sample_many(convs[i : i + 20], n=n, thinking=True, max_tokens=8192)
    with OUT.open("a") as f:
        for (p, c), samples in zip(cells[i : i + 20], outs):
            for k, s in enumerate(samples):
                f.write(json.dumps({"probe": p["id"], "cond": c, "k": k, **s}, ensure_ascii=False) + "\n")
    print(f"done {min(i + 20, len(cells))}/{len(cells)}", flush=True)
