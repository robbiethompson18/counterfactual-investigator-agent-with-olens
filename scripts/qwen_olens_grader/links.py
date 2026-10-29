# id -> docent link (same rules as scripts/docent_upload.py link_for) + env viewer expview link for run-folder episodes
import collections
import json
import sys
from pathlib import Path

ROOT = Path("/home/claude/workspace/counterfactual-investigator-agent-with-olens/data/runs")
ENVS = ROOT.parent / "envs"
WEB = "https://docent.transluce.org"
EV = "https://robbiethompson18.github.io/counterfactual-investigator-agent-with-olens/expview/"
_idx = {}


def link_for(source):
    source = source.split("|")[0]
    run, _, line = source.rpartition(":")
    ep = ENVS / run / "episodes"
    if ep.is_dir():
        if run not in _idx:
            _idx[run] = json.loads((ep / "index.json").read_text())
        index = _idx[run]
        key = line if line in index else (list(index)[int(line)] if line.isdigit() else None)
        if key is None:
            return None, None
        shard, off, n = index[key]
        with open(ep / shard, "rb") as f:
            f.seek(off)
            d = json.loads(f.read(n))
        env, _, r = run.partition("/")
        return d.get("docent_url"), f"{EV}#env={env}&run={r}&episode={key}"
    p = ROOT / run / "docent.json"
    if not p.exists():
        return None, None
    if run not in _idx:
        _idx[run] = json.loads(p.read_text())
    m = _idx[run]
    try:
        rid = m["runs"][int(line)]
    except (json.JSONDecodeError, KeyError):
        return None, None
    return (rid and f"{WEB}/dashboard/{m['collection_id']}/agent_run/{rid}"), None


if __name__ == "__main__":
    H = [json.loads(l) for l in Path(f"{sys.argv[1]}/hits3.jsonl").read_text().splitlines()]
    ids = sorted({h["id"] for h in H})
    c = collections.Counter()
    out = {}
    for i in ids:
        try:
            d, e = link_for(i)
        except (json.JSONDecodeError, KeyError):
            d, e = None, None
        out[i] = [d, e]
        c[(i.split("/")[0], bool(d))] += 1
    Path(f"{sys.argv[1]}/links.json").write_text(json.dumps(out))
    envs = sorted({k[0] for k in c})
    print({e: (c[(e, True)], c[(e, False)]) for e in envs})
