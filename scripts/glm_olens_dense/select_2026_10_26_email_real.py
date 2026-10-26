"""2026-10-26: the six Claude Code-harness email_ambiguous_real dirs; quarter per arm (seed 0), punct + template, no whitespace-only,
canonical shared prefixes of sampled episodes kept."""

import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

out, jobs = Path("sel_real"), Path("jobs_real")
out.mkdir(exist_ok=True)
jobs.mkdir(exist_ok=True)
tot = 0
for name in [
    "real_2026-10-25_glm53",
    "real_2026-10-25_glm53_corp16",
    "real_2026-10-25_glm53_acme48",
    "real_2026-10-25_glm53_junebug16",
    "real_2026-10-25_glm53_junebug48",
    "real_2026-10-25_glm53_tool64",
]:
    prep = Path("prep") / name
    P = pd.read_parquet(prep / "positions.parquet", columns=["tid", "pos", "row_index", "token", "is_special"])
    E = pd.read_parquet(prep / "episodes.parquet", columns=["tid", "row_index", "arm"])
    rng = random.Random(0)
    keep_rows = set()
    for _, g in E.drop_duplicates("row_index").sort_values("row_index").groupby("arm", sort=True):
        ix = list(g.row_index)
        rng.shuffle(ix)
        keep_rows |= set(ix[: len(ix) // 4])
    D = pd.read_parquet(prep / "dense_prefix_dups.parquet")
    kt = set(E[E.row_index.isin(keep_rows)].tid)
    need = set(zip(D[D.tid.isin(kt)].canon_tid, D[D.tid.isin(kt)].canon_pos.astype(int)))
    sel = P[P.row_index.isin(keep_rows).to_numpy() | np.array([(t, int(p)) in need for t, p in zip(P.tid, P.pos)], dtype=bool)]
    sel = sel[((sel.token.str.strip() != "") | sel.is_special).to_numpy()]
    sel[["tid", "pos"]].to_parquet(out / f"{name}.parquet", index=False)
    (out / f"{name}_tids.json").write_text(json.dumps(sorted(kt)))
    views = {v["tid"]: v for v in pickle.loads((prep / "views.pkl").read_bytes())}
    g = sel.groupby("tid").pos.apply(lambda s: sorted(int(x) for x in s))
    tids = list(g.index)
    k, cur, n = 0, [], 0
    for t in tids + [None]:
        if t is not None:
            cur.append(t)
            n += len(g[t])
        if cur and (t is None or n >= 20000):
            job = {"ids": [views[x]["ids"][: max(g[x]) + 1].tolist() for x in cur], "pos": [g[x] for x in cur], "tids": cur, "seed": 7 + k}
            torch.save(job, jobs / f"{name}_v2__{k:04d}.job.pt")
            k += 1
            cur, n = [], 0
    tot += len(sel)
    print(name, "episodes", len(keep_rows), "of", E.row_index.nunique(), "positions", len(sel), "jobs", k, flush=True)
print("TOTAL", tot)
