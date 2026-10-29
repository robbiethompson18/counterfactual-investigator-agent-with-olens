import collections
import json
import sys

sys.path.insert(0, sys.argv[1])
from pathlib import Path

import links as L

SEL = {}
for l in (
    Path(
        "/home/claude/workspace/counterfactual-investigator-agent-with-olens/data/runs/donation_bet/2026-10-27_olens-nothink/selection.jsonl"
    )
    .read_text()
    .splitlines()
):
    d = json.loads(l)
    SEL[d["id"]] = f"donation_bet/{d['run']}:{d['episode']}"


def resolve(i):
    i = i.split("|")[0]
    if i.startswith("qwen/") and i in SEL:
        return L.link_for(SEL[i])
    if i.startswith("alignment_faking/2026-10-28_qwen36"):
        L.ENVS = Path("/home/claude/workspace/cia-af-qwen36/data/envs")
        try:
            return L.link_for(i)
        finally:
            L.ENVS = L.ROOT.parent / "envs"
    return L.link_for(i)


H = [json.loads(l) for l in Path(f"{sys.argv[1]}/hits3.jsonl").read_text().splitlines()]
out = {}
c = collections.Counter()
for i in sorted({h["id"] for h in H}):
    try:
        out[i] = list(resolve(i))
    except (json.JSONDecodeError, KeyError):
        out[i] = [None, None]
    c[(i.split("/")[0], bool(out[i][0]))] += 1
Path(f"{sys.argv[1]}/links.json").write_text(json.dumps(out))
print({k: v for k, v in c.items() if not k[1]})
