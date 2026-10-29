# pass 2: SIT regex over every distinct (prefix, layer) Qwen O-lens read; keep full prefix for hits; per-group episode sets
import collections
import glob
import hashlib
import json
import os
import re
import sys
from pathlib import Path

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, S)
from sit import SIT


def parse(i):
    env, _, rest = i.partition("/")
    run, _, ep = rest.partition(":")
    return env, run, re.split(r"[~|]", ep)[0]


files = Path(f"{S}/files.txt").read_text().split() + sorted(
    glob.glob("/home/claude/workspace/tmp/qwen_hf_reads/runs/olens_reads/*/reads*.jsonl")
)
files = [f for f in files if "transcripts" not in f]
seen = set()
eps = collections.defaultdict(set)
npos = collections.Counter()
out = Path(f"{S}/hits2.jsonl").open("w")  # noqa: SIM115  (streamed for the whole scan)
for path in files:
    src = path.replace("/home/claude/workspace/", "")
    with open(path) as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except (json.JSONDecodeError, KeyError):
                continue
            pre = d.get("prefix") or ""
            ph = hashlib.md5(pre.encode()).digest()
            key = (ph, d.get("layer"))
            if key in seen:
                continue
            seen.add(key)
            i = d.get("id", "")
            env, run, arm = parse(i)
            g = (env, run, arm)
            eps[g].add(i)
            npos[g] += 1
            for bi, x in enumerate(d.get("bullets") or []):
                t = x["text"] if isinstance(x, dict) else str(x)
                m = SIT.search(t)
                if not m:
                    continue
                out.write(
                    json.dumps(
                        {
                            "src": src,
                            "id": i,
                            "env": env,
                            "run": run,
                            "arm": arm,
                            "layer": d.get("layer"),
                            "token_index": d.get("token_index"),
                            "token": d.get("token"),
                            "role": d.get("role"),
                            "msg_index": d.get("msg_index"),
                            "bullet": bi,
                            "text": t,
                            "match": m.group(0),
                            "echo": bool(SIT.search(pre)),
                            "prefix": pre,
                        }
                    )
                    + "\n"
                )
    print("done", src, len(seen), flush=True)
Path(f"{S}/groups2.json").write_text(
    json.dumps([{"env": k[0], "run": k[1], "arm": k[2], "episodes": len(v), "positions": npos[k]} for k, v in eps.items()])
)
print("positions", len(seen))
