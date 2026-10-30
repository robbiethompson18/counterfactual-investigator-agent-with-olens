"""Kylie's 2026-10-27 23:24 selection over the prep_envs.py fill: "just make sure 64 in each which show the behaviour have them".

Keeps, from the not-yet-run prep_envs jobs, only episodes with PRIMARY (`positive`) true in their run's latest measure labels, up to
64 per (run, arm) chosen at seed 0, and in them every template/special token plus punctuation on the simple tasks (over-eagerness
email*, alignment_faking, essay_grading, hinted_truthfulqa). Jobs already done are kept as they are.

    uv run --with torch python scripts/glm_olens_dense/select_2026_10_27_positive.py --fill <dir with prep/ jobs/ out_jobs/> \
        [--running p0__0003.job.pt]

Writes <fill>/jobs_pos/q__NNNN.job.pt and <fill>/selection_pos64.csv (per run: episodes, arms, new template / punctuation positions).
"""

import argparse
import glob
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

SIMPLE = {"alignment_faking", "essay_grading", "hinted_truthfulqa", "vulnerable_users"}


def labels(run_dir: str) -> dict:
    files = sorted(Path("data", run_dir, "labels").glob("measure@*.jsonl"), key=lambda p: p.stat().st_mtime)
    return {r["episode"]: r["values"] for r in map(json.loads, files[-1].read_text().splitlines())} if files else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", type=Path, required=True)
    ap.add_argument("--running", default="", help="job the worker is on now (kept, not re-split)")
    ap.add_argument("--k", type=int, default=64)
    a = ap.parse_args()
    eps, special = [], {}
    for d in sorted((a.fill / "prep").iterdir()):
        if not (d / "episodes.parquet").exists():
            continue
        E = pd.read_parquet(d / "episodes.parquet", columns=["tid", "run_dir", "episode_id", "arm"])
        lab = labels(E.run_dir.iloc[0])
        eps.append(E.assign(positive=[(lab.get(i) or {}).get("positive") is True for i in E.episode_id]))
        P = pd.read_parquet(d / "positions.parquet", columns=["tid", "pos", "is_special"])
        special.update({(t, int(p)): s for t, p, s in zip(P.tid, P.pos, P.is_special)})
    E = pd.concat(eps)
    E["env"], E["task"] = E.run_dir.str.split("/").str[1], E.arm.str.split("/").str[0]
    pick = set()
    for _, g in E[E.positive].groupby(["run_dir", "arm"]):
        ids = sorted(g.tid)
        pick |= set(ids if len(ids) <= a.k else random.Random(0).sample(ids, a.k))
    info = E.set_index("tid")
    done = {Path(f).name.replace(".out.pt", ".job.pt") for f in glob.glob(str(a.fill / "out_jobs/*.out.pt"))}
    out = a.fill / "jobs_pos"
    out.mkdir(exist_ok=True)
    views, stats = [], defaultdict(lambda: [0, 0])
    for f in sorted(glob.glob(str(a.fill / "jobs/p*.job.pt"))):
        if Path(f).name in done | {a.running}:
            continue
        j = torch.load(f)
        for ids, ps, t in zip(j["ids"], j["pos"], j["tids"]):
            if t not in pick:
                continue
            r = info.loc[t]
            simple = r.env in SIMPLE or (r.env == "overeagerness" and r.task.startswith("email"))
            keep = [p for p in ps if special[(t, p)] or simple]
            if keep:
                views.append((ids[: max(keep) + 1], keep, t))
                for p in keep:
                    stats[r.run_dir][0 if special[(t, p)] else 1] += 1
    k, cur, n = 0, [], 0
    for i, v in enumerate(views):
        cur.append(v)
        n += len(v[1])
        if n >= 20000 or i == len(views) - 1:
            job = {"ids": [x[0] for x in cur], "pos": [x[1] for x in cur], "tids": [x[2] for x in cur], "seed": (900_000 + k) * 10_000_019}
            torch.save(job, out / f"q__{k:04d}.job.tmp")
            (out / f"q__{k:04d}.job.tmp").rename(out / f"q__{k:04d}.job.pt")
            k, cur, n = k + 1, [], 0
    S = E[E.tid.isin(pick)].groupby("run_dir").agg(episodes=("tid", "size"), arms=("arm", "nunique"))
    S = S.join(pd.DataFrame(stats, index=["template", "punct"]).T).fillna(0)
    S.to_csv(a.fill / "selection_pos64.csv")
    print(S.to_string())
    print(f"{k} jobs, {int(S.template.sum() + S.punct.sum()):,} positions ({int(S.template.sum()):,} template)")


if __name__ == "__main__":
    main()
