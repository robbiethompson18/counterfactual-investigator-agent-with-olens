"""Does the O-lens, read early in the CoT (before any creator is named), predict whether the answer will say
Google or Alibaba? Reads at the first K newlines of the reasoning for Google-answer vs Alibaba-answer samples.

    uv run ... python scripts/identity/env_google/lens_cot.py probe_f supportfn:l direct 8
"""

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, "scripts/identity")
sys.path.insert(0, "scripts/identity/env_google")
from envs import build
from probe import CREATORS, PROBES, self_claims
from qwen import read, render

src, cond, probe, per = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rows = [json.loads(line) for line in open(f"data/runs/identity/2026-10-21_env_google/{src}.jsonl")]
rows = [r for r in rows if r["cond"] == cond and r["probe"] == probe]
anyname = re.compile("|".join(CREATORS.values()), re.IGNORECASE)
out = []
for lab in ("google", "alibaba"):
    pool = [r for r in rows if r["ans"] == lab]
    random.Random(0).shuffle(pool)
    for r in pool[:per]:
        msgs = build(cond, r["seed"]) + [{"role": "user", "content": PROBES[probe]}]
        ids, toks = render(msgs, {"reasoning": r["reasoning"], "content": r["content"]})
        think = max(i for i, t in enumerate(toks) if t == "<think>")
        cot_text, pos = "", []
        for i in range(think + 1, len(toks)):
            cot_text += toks[i]
            if anyname.search(cot_text):
                break
            if "\n" in toks[i] and cot_text.strip():
                pos.append(i)
            if len(pos) == 2:
                break
        items = [(p, L) for p in pos for L in (40, 48, 56)]
        res = read(ids, items) if items else []
        for (p, L), rr in zip(items, res):
            x = rr[0]
            out.append(
                {
                    "ans": lab,
                    "seed": r["seed"],
                    "pos": p,
                    "layer": L,
                    "cot_prefix": "".join(toks[think + 1 : p + 1]),
                    "bullets": x["bullets"],
                    "bfve": x["bfve"],
                    "claims": sorted(set().union(*[self_claims(b) for b in x["bullets"]])),
                }
            )
Path(f"data/runs/identity/2026-10-21_env_google/lens_cot_{cond.replace(':', '_')}.jsonl").write_text(
    "".join(json.dumps(o) + "\n" for o in out)
)
for lab in ("google", "alibaba"):
    s = [o for o in out if o["ans"] == lab]
    print(
        lab,
        "answers: reads",
        len(s),
        "google_claim",
        sum("google" in o["claims"] for o in s),
        "alibaba_claim",
        sum("alibaba" in o["claims"] for o in s),
        "any-google-word",
        sum(bool(re.search(CREATORS["google"], " ".join(o["bullets"]))) for o in s),
    )
