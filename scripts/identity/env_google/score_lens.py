"""Share of O-lens readouts with a positive Google / Alibaba self-claim, per condition (and per layer)."""

import json
import sys
from collections import defaultdict

sys.path.insert(0, "scripts/identity/env_google")
from probe import self_claims

for path in sys.argv[1:]:
    agg = defaultdict(lambda: [0, 0, 0])
    for line in open(path):
        r = json.loads(line)
        c = set().union(*[self_claims(b) for b in r["bullets"]])
        for key in (r["cond"], f"{r['cond']}@L{r['layer']}"):
            a = agg[key]
            a[0] += 1
            a[1] += "google" in c
            a[2] += "alibaba" in c
    print("==", path)
    for k, (n, g, al) in agg.items():
        if "@L" not in k:
            print(f"{k[:28]:28s} n={n:3d} google_claim={g:3d} ({g / n:.0%}) alibaba_claim={al:3d} ({al / n:.0%})")
