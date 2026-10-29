import collections
import glob
import json
import sys
from pathlib import Path

S = sys.argv[1]
inp = {}
out = {}
for b in sorted(glob.glob(f"{S}/triage/b*")):
    for l in Path(f"{b}/input.jsonl").read_text().splitlines():
        d = json.loads(l)
        inp[d["key"]] = d | {"batch": b[-2:]}
    for l in Path(f"{b}/output.jsonl").read_text().splitlines():
        try:
            d = json.loads(l)
            out[d["key"]] = d
        except (json.JSONDecodeError, KeyError):
            print("bad", b, l[:80])
print(len(inp), len(out), len(set(inp) - set(out)))
c = collections.defaultdict(collections.Counter)
for k, i in inp.items():
    o = out.get(k, {})
    c[i["env"]][o.get("label", "?")] += 1
    if o.get("label") == "eval_aware":
        c[i["env"]][f"s{o.get('strength', 0)}"] += 1
L = ["eval_aware", "user_testing", "fiction", "echo", "other", "s3", "s2"]
print(f"{'env':36}" + "".join(f"{x:>13}" for x in L))
for e in sorted(c, key=lambda e: -c[e]["eval_aware"]):
    print(f"{e[:36]:36}" + "".join(f"{c[e][x]:13}" for x in L))
Path(f"{S}/triaged.json").write_text(json.dumps({k: inp[k] | {"judge": out.get(k)} for k in inp}))
