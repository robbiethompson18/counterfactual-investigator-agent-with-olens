"""Kylie's 2026-10-27 23:46 cut of the 64-positive selection (select_2026_10_27_positive.py): over-eagerness only on the arms of
the env's "main: one-edit fixes and nulls" COMPARISONS (payback, dana, calendar, gitbranch, email junebug + 4 Johns); other envs
as queued. Re-splits the not-yet-run q__ jobs into <fill>/jobs_main/r__NNNN.job.pt.

    uv run --with torch python scripts/glm_olens_dense/select_2026_10_27_main.py <job the worker is on now>
"""

import glob
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

F = Path.home() / "workspace/tmp/glm_fill"
ND = ["no_criterion", "nc_done_plain", "nc_done_home", "nc_done_neutral"]
COMP = {
    "payback: base vs one edit": [f"payback/{v}" for v in ["base", "no_done", "base_plain", "base_home", "base_done_neutral"]],
    "dana: criterion-free base vs one edit": [f"dana/{v}" for v in ["base_no_criterion", *ND]],
    "calendar: criterion-free base vs one edit": [f"calendar/{v}" for v in ["base_no_criterion", *ND]],
    "gitbranch: criterion-free base vs one edit": [f"gitbranch/{v}" for v in ["base_no_criterion", *ND]],
    "email (junebug) + 4 Johns: base vs single edits": ["email/fixed@lowc"]
    + [f"email/{v}@lowc" for v in ["no_done", "fixed_plain", "fixed_home", "base_done_neutral"]],
}
ARM2C = {a: c for c, arms in COMP.items() for a in arms}
E = pd.concat(
    pd.read_parquet(d / "episodes.parquet", columns=["tid", "run_dir", "episode_id", "arm", "item"])
    for d in (F / "prep").iterdir()
    if (d / "episodes.parquet").exists()
).set_index("tid")
special = {}
for d in (F / "prep").iterdir():
    if d.name.startswith("overeagerness") and (d / "positions.parquet").exists():
        P = pd.read_parquet(d / "positions.parquet", columns=["tid", "pos", "is_special"])
        special.update({(t, int(p)): s for t, p, s in zip(P.tid, P.pos, P.is_special)})
done = {Path(f).name.replace(".out.pt", ".job.pt") for f in glob.glob(str(F / "out_jobs/*.out.pt"))}
running = sys.argv[1]
out = F / "jobs_main"
out.mkdir(exist_ok=True)
for f in out.glob("*"):
    f.unlink()
views, st = [], defaultdict(lambda: {"episodes": set(), "template": 0, "punct": 0})
for f in sorted(glob.glob(str(F / "jobs_pos/q__*.job.pt"))):
    if Path(f).name in done or Path(f).name == running:
        continue
    j = torch.load(f)
    for ids, ps, t in zip(j["ids"], j["pos"], j["tids"]):
        r = E.loc[t]
        if r.run_dir.startswith("envs/overeagerness/"):
            if r.arm not in ARM2C:
                continue
            key = (ARM2C[r.arm] if not (r.arm.startswith("email") and str(r["item"]) == "0") else "4 Johns (item 0)", r.arm)
        else:
            key = (r.run_dir.split("/")[1], "")
        views.append((ids, ps, t))
        s = st[key]
        s["episodes"].add(t)
        if r.run_dir.startswith("envs/overeagerness/"):
            s["template"] += sum(special[(t, p)] for p in ps)
            s["punct"] += sum(not special[(t, p)] for p in ps)
        else:
            s["template"] += len(ps)  # non-overeagerness: total positions (split not tracked here)
k, cur, n = 0, [], 0
for i, v in enumerate(views):
    cur.append(v)
    n += len(v[1])
    if n >= 20000 or i == len(views) - 1:
        torch.save(
            {"ids": [x[0] for x in cur], "pos": [x[1] for x in cur], "tids": [x[2] for x in cur], "seed": (800_000 + k) * 10_000_019},
            out / f"r__{k:04d}.job.tmp",
        )
        (out / f"r__{k:04d}.job.tmp").rename(out / f"r__{k:04d}.job.pt")
        k, cur, n = k + 1, [], 0
rows = [
    {"comparison": c, "arm": a, "episodes": len(s["episodes"]), "template": s["template"], "punct": s["punct"]}
    for (c, a), s in sorted(st.items())
]
T = pd.DataFrame(rows)
T.to_csv(F / "selection_main.csv", index=False)
print(T.to_string())
print("jobs", k, "positions", sum(len(v[1]) for v in views))
