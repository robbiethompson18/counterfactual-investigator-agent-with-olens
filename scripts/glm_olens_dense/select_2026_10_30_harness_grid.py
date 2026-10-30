"""Kylie's 2026-10-30 harness_grid O-lens cut: every prompt position (p0) for all GLM episodes, plus full-episode reads (p1) for
K episodes per (task x wording x thinking mode), balanced: half acted without asking (Haiku asked_judge says no ask), half
asked, each pick taking the harness-factor combination least used so far in its cell.

    uv run python scripts/glm_olens_dense/select_2026_10_30_harness_grid.py --k 3 --out p1_select.json
"""

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from environments._lib.index import latest_labels
from environments._lib.runfolder import read_episodes
from environments.harness_grid.env import FACTORS

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--runs", default="data/envs/harness_grid/*_glm53sh-full-think*")
ap.add_argument("--out", required=True)
a = ap.parse_args()
cells = collections.defaultdict(lambda: {"acted": [], "asked": []})
for d in sorted(Path(".").glob(a.runs)):
    _, m = latest_labels(d)
    hk = {}
    for f in (d / "labels").glob("asked_judge@*.jsonl"):
        hk |= {x["episode"]: x["values"].get("asks") for x in map(json.loads, f.read_text().splitlines())}
    mode = "off" if "thinkoff" in d.name else "on"
    for e in read_episodes(d):
        v = m.get(e["id"]) or {}
        if "positive" not in v or e["id"] not in hk:
            continue
        task, _, var = e["arm"].partition("/")
        f = dict(zip(["wording", *FACTORS], var.split(".")))
        kind = "acted" if v["positive"] and not hk[e["id"]] else "asked" if not v["positive"] and hk[e["id"]] else None
        if kind:
            cells[(task, f["wording"], mode)][kind].append((f"envs/harness_grid/{d.name}/{e['id']}", tuple(f[k] for k in FACTORS)))
rng = random.Random(20261030)
pick = []
for key, c in sorted(cells.items()):
    used = collections.Counter()
    want = {"acted": a.k // 2 + a.k % 2, "asked": a.k // 2}
    if len(c["acted"]) < want["acted"]:  # top up from the other kind when one is rare
        want["asked"] += want["acted"] - len(c["acted"])
    for kind in ("acted", "asked"):
        pool = c[kind][:]
        rng.shuffle(pool)
        for _ in range(min(want[kind], len(pool))):
            best = min(pool, key=lambda x: sum(used[(i, v)] for i, v in enumerate(x[1])))
            pool.remove(best)
            pick.append(best[0])
            used.update(enumerate(best[1]))
Path(a.out).write_text(json.dumps(pick))
by = collections.Counter(("off" if "thinkoff" in p else "on") for p in pick)
print(len(pick), "episodes", dict(by), "cells", len(cells))
