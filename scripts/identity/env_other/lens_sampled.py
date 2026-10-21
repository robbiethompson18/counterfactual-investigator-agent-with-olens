"""Sampled (T=1) O-lens reads at the direct-probe header, to compare creator shares between envs whose answers differ.
usage: python lens_sampled.py env1,env2,... [n] [layers]"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts/identity")
sys.path.insert(0, "scripts/identity/env_other")
from classify import first, mentions
from envs import DIRECT
from qwen import header_positions, read, render

OUT = Path("data/runs/identity/2026-10-21_env_other")
names = sys.argv[1].split(",")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
LAYERS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [36, 44]
envs = {e["name"]: e for e in json.loads((OUT / "filled_envs.json").read_text())}
with (OUT / "lens_sampled.jsonl").open("a") as f:
    for name in names:
        e = envs[name]
        ids, toks = render(e["messages"] + [{"role": "user", "content": DIRECT[e["lang"]]}])
        items = [(p, L) for p in header_positions(ids) if toks[p] != "\n" for L in LAYERS]
        res = read(ids, items, temperature=1.0, n=N, seed=1)
        bl = []
        for (p, L), rr in zip(items, res):
            for ro in rr:
                f.write(json.dumps({"env": name, "pos": p, "tok": toks[p], "layer": L, **ro}, ensure_ascii=False) + "\n")
                bl += list(zip(ro["bullets"], ro["bfve"]))
        c1 = Counter(first(b) for b, _ in bl)
        ca = Counter(k for b, _ in bl for k in mentions(b))
        print(f"{name:20s} bullets={len(bl)} first={dict(c1.most_common())} any={dict(ca.most_common())}", flush=True)
