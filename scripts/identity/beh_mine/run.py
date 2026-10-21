"""Sample probes x conditions (thinking on), label them by regex (no LLM grader:
free-text probes get a normalized raw answer). CoT identity mentions are graded later by hand / subagents. Appends to data/runs/identity/2026-10-21_beh_mine/raw_<tag>.jsonl.

uv run --with openai --with anthropic --with transformers --with jinja2 python scripts/identity/beh_mine/run.py <tag> <n> [probe,probe|all] [cond,cond]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts/identity")
sys.path.insert(0, "scripts/identity/beh_mine")
from probes import CONDITIONS, CUED, PROBES, VENDOR_RE, score, vendors_in
from qwen import sample_many

OUT = Path("data/runs/identity/2026-10-21_beh_mine")
tag, n = sys.argv[1], int(sys.argv[2])
probes = list(PROBES) if len(sys.argv) < 4 or sys.argv[3] == "all" else sys.argv[3].split(",")
conds = sys.argv[4].split(",") if len(sys.argv) > 4 else list(CONDITIONS)


def msgs(cond, probe):
    s = CONDITIONS[cond]
    return ([{"role": "system", "content": s}] if s else []) + [{"role": "user", "content": PROBES[probe][0]}]


cells = [(p, c) for p in probes for c in conds]
outs = sample_many([msgs(c, p) for p, c in cells], n=n, thinking=True, max_tokens=6144)
rows = [{"probe": p, "cond": c, **s} for (p, c), ss in zip(cells, outs) for s in ss]
for r in rows:
    r["label"] = score(r["probe"], r["content"], r["reasoning"])
    r["answer_vendors"] = vendors_in(r["content"])
    r["cot_vendors"] = vendors_in(r["reasoning"])
    r["cot_names_cued"] = bool(VENDOR_RE[CUED[r["cond"]]].search(r["reasoning"] or ""))

OUT.mkdir(parents=True, exist_ok=True)
with (OUT / f"raw_{tag}.jsonl").open("a") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("wrote", len(rows), "rows to", OUT / f"raw_{tag}.jsonl")
