"""Targeted O-lens read: the newline after `assistant` in the final header (where Google self-claims showed up at
L56), layers 52/56/60, n sampled readouts at T=0.7. Bullet-level Google / Alibaba self-claim rates per condition.

    uv run ... python scripts/identity/env_google/lens_nl.py OUTNAME cond1 cond2 ...
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts/identity")
sys.path.insert(0, "scripts/identity/env_google")
from envs import build
from probe import PROBES, self_claims
from qwen import header_positions, read, render

out, conds = sys.argv[1], sys.argv[2:]
probe = "direct"
rows = []
for c in conds:
    ids, toks = render(build(c, 0) + [{"role": "user", "content": PROBES[probe]}])
    pos = header_positions(ids)[-2]
    assert toks[pos] == "\n" and toks[pos - 1] == "assistant", toks[pos - 2 : pos + 1]
    items = [(pos, L) for L in (52, 56, 60)]
    for (p, L), res in zip(items, read(ids, items, temperature=0.7, n=8)):
        for x in res:
            for b, f in zip(x["bullets"], x["bfve"]):
                rows.append({"cond": c, "layer": L, "bullet": b, "bfve": f, "claims": sorted(self_claims(b))})
    for L in (52, 56, 60, "all"):
        s = [r for r in rows if r["cond"] == c and (L == "all" or r["layer"] == L)]
        g, a = sum("google" in r["claims"] for r in s), sum("alibaba" in r["claims"] for r in s)
        print(f"{c:24s} L{L}: bullets={len(s)} google={g} alibaba={a}", flush=True)
Path(f"data/runs/identity/2026-10-21_env_google/{out}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
