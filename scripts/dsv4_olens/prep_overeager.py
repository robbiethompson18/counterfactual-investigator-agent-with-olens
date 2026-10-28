"""requests.jsonl for capture_env.py from environments/ runs (data/envs/<env>/<run>): --per-arm seeded random episodes of each
arm (pooled over the runs), messages and tools as the model saw them. Ids are `ENV/RUN:<episode id>`, so the reads attach
to their episodes in expview.

    python scripts/dsv4_olens/prep_overeager.py --env overeagerness --runs 2026-10-27_dsv4flash-main256 \
        --arms payback/base,... --per-arm 64 --out data/dsv4_olens/overeager/requests.jsonl
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from environments._lib.index import latest_labels
from environments._lib.runfolder import LOCAL, read_episodes

ap = argparse.ArgumentParser()
ap.add_argument("--env", required=True)
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--arms", required=True, help="comma list")
ap.add_argument("--items", default=None, help="comma list of items to keep")
ap.add_argument("--per-arm", type=int, default=64)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

arms = a.arms.split(",")
pool: dict[str, list] = {x: [] for x in arms}
for run in a.runs:
    d = LOCAL / a.env / run
    _, labels = latest_labels(d)
    for ep in read_episodes(d):
        v = labels.get(ep["id"]) or {}
        if ep["arm"] not in pool or ep.get("error") or v.get("positive") is None or (a.items and ep["item"] not in a.items.split(",")):
            continue
        pool[ep["arm"]].append((run, ep, v))
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    for arm in arms:
        eps = sorted(pool[arm], key=lambda x: (x[0], x[1]["id"]))
        random.Random(f"{a.seed}:{arm}").shuffle(eps)
        for run, ep, v in eps[: a.per_arm]:
            req = {"id": f"{a.env}/{run}:{ep['id']}", "env": a.env, "arm": arm, "item": ep["item"], "positive": v["positive"]}
            req |= {"asked": v.get("asked"), "thinking": True, "messages": ep["messages"], "tools": ep["tools"]}
            f.write(json.dumps(req) + "\n")
        print(f"{arm}: {min(len(eps), a.per_arm)} of {len(eps)}")
