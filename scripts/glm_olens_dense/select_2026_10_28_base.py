"""Kylie 2026-10-28 00:00: over-eagerness only on the BASE arm of each main comparison (payback/base, dana/calendar/gitbranch
base_no_criterion, email/fixed@lowc), other envs after. Splits the not-yet-run r__ jobs (select_2026_10_27_main.py) into
<fill>/jobs_base/t__NNNN.job.pt and <fill>/jobs_rest/u__NNNN.job.pt.

    uv run --with torch python scripts/glm_olens_dense/select_2026_10_28_base.py <job the worker is on now>
"""

import glob
import sys
from pathlib import Path

import pandas as pd
import torch

F = Path.home() / "workspace/tmp/glm_fill"
BASE = {"payback/base", "dana/base_no_criterion", "calendar/base_no_criterion", "gitbranch/base_no_criterion", "email/fixed@lowc"}
E = pd.concat(
    pd.read_parquet(d / "episodes.parquet", columns=["tid", "run_dir", "arm", "item"])
    for d in (F / "prep").iterdir()
    if (d / "episodes.parquet").exists()
).set_index("tid")
done = {Path(f).name.replace(".out.pt", ".job.pt") for f in glob.glob(str(F / "out_jobs/*.out.pt"))}
skip = set(sys.argv[1:])
base, rest = [], []
for f in sorted(glob.glob(str(F / "jobs_main/r__*.job.pt"))):
    if Path(f).name in done or Path(f).name in skip:
        continue
    j = torch.load(f)
    for v in zip(j["ids"], j["pos"], j["tids"]):
        r = E.loc[v[2]]
        if r.run_dir.startswith("envs/overeagerness/"):
            if r.arm in BASE:
                base.append(v)
        else:
            rest.append(v)


def write(views, prefix, out):
    out.mkdir(exist_ok=True)
    [x.unlink() for x in out.glob("*")]
    k, cur, n = 0, [], 0
    for i, v in enumerate(views):
        cur.append(v)
        n += len(v[1])
        if n >= 20000 or i == len(views) - 1:
            torch.save(
                {
                    "ids": [x[0] for x in cur],
                    "pos": [x[1] for x in cur],
                    "tids": [x[2] for x in cur],
                    "seed": (hash(prefix) % 1000 * 1000 + k) * 10_000_019,
                },
                out / f"{prefix}__{k:04d}.job.tmp",
            )
            (out / f"{prefix}__{k:04d}.job.tmp").rename(out / f"{prefix}__{k:04d}.job.pt")
            k, cur, n = k + 1, [], 0
    return k


print("base", write(base, "t", F / "jobs_base"), sum(len(v[1]) for v in base), "eps", len(base))
print("rest", write(rest, "u", F / "jobs_rest"), sum(len(v[1]) for v in rest), "eps", len(rest))
