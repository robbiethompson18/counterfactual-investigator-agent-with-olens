"""O-lens reads at the final assistant header for a condition's contexts, final user turn = a probe
(default: the behaviour question, which never mentions identity). Counts readouts/bullets naming each creator.

    uv run ... python scripts/identity/env_google/lens.py OUTNAME --probe behaviour --seeds 8 cond1 cond2 ...
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts/identity")
sys.path.insert(0, "scripts/identity/env_google")
from envs import build
from probe import PROBES, creators
from qwen import header_positions, read, render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("conds", nargs="+")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--probe", default="behaviour")
    ap.add_argument("--layers", default="32,40,48,56")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    rows = []
    for c in a.conds:
        for s in range(a.seeds):
            msgs = build(c, s) + [{"role": "user", "content": PROBES[a.probe]}]
            ids, toks = render(msgs)
            pos = header_positions(ids)
            res = read(ids, [(p, L) for p in pos for L in layers], temperature=a.temperature, n=a.n)
            for (p, L), r in zip([(p, L) for p in pos for L in layers], res):
                for x in r:
                    rows.append(
                        {
                            "cond": c,
                            "seed": s,
                            "pos": p,
                            "tok": toks[p],
                            "layer": L,
                            "text": x["text"],
                            "bullets": x["bullets"],
                            "bfve": x["bfve"],
                        }
                    )
        cs = [r for r in rows if r["cond"] == c]
        cnt = Counter(k for r in cs for k in creators(r["text"]))
        print(f"{c:22s} reads={len(cs)} readouts naming: {dict(cnt)}", flush=True)
    Path(f"data/runs/identity/2026-10-21_env_google/{a.out}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


main()
