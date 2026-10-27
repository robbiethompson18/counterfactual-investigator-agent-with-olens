"""Kylie's 23:07 rules: half the episodes per (arm, item) (seed 0), punctuation + template tokens, no whitespace-only tokens;
Donation Bet: the given 168-row sample, same rule + the first 6 reply tokens."""

import json
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

FIRST_ADDED, THINK_END = 154820, None
out = Path("sel_v2")
out.mkdir(exist_ok=True)
jobs = Path("jobs_v2")
jobs.mkdir(exist_ok=True)
done = torch.load("jobs_late/2026-10-25_glm53_email_ambiguous_part8_fresh128__0000.job.pt")  # running on the box under the old rule
done_set = {(t, int(p)) for t, ps in zip(done["tids"], done["pos"]) for p in ps}
tot = {}
for name, src, half, first6 in [
    ("2026-10-25_glm53_email_ambiguous_part8_fresh128", "email_ambiguous/2026-10-25_glm53_email_ambiguous_part8_fresh128", True, False),
    ("dbet_nothink", None, False, True),
    ("2026-10-25_glm53_email_ambiguous_part4_corp", "", True, False),
    ("2026-10-25_glm53_email_ambiguous_part5_junebug", "", True, False),
    ("2026-10-25_glm53_email_ambiguous_part6_junebug_topup", "", True, False),
    ("2026-10-25_glm53_email_ambiguous_part7_acme_and_128", "", True, False),
]:
    prep = Path("prep") / name
    P = pd.read_parquet(prep / "positions.parquet", columns=["tid", "pos", "row_index", "arm", "item", "token", "is_special"])
    E = pd.read_parquet(prep / "episodes.parquet", columns=["tid", "row_index", "arm", "item"])
    if half:
        rng = random.Random(0)
        keep_rows = set()
        for _, g in E.drop_duplicates("row_index").sort_values("row_index").groupby("arm", sort=True):
            ix = list(g.row_index)
            rng.shuffle(ix)
            keep_rows |= set(ix[: len(ix) // 2])
    else:
        keep_rows = set(E.row_index)
    views = {v["tid"]: v for v in pickle.loads((prep / "views.pkl").read_bytes())}
    early = set()
    if first6:
        tend = 154842  # </think>
        for t, v in views.items():
            ids = v["ids"]
            w = np.where(ids == tend)[0]
            if len(w):
                early |= {(t, int(p)) for p in range(w[-1] + 1, min(w[-1] + 7, len(ids)))}
    D = pd.read_parquet(prep / "dense_prefix_dups.parquet")
    kt = set(E[E.row_index.isin(keep_rows)].tid)
    need = set(zip(D[D.tid.isin(kt)].canon_tid, D[D.tid.isin(kt)].canon_pos.astype(int)))  # shared prefixes read in another episode
    sel = P[P.row_index.isin(keep_rows).to_numpy() | np.array([(t, int(p)) in need for t, p in zip(P.tid, P.pos)], dtype=bool)]
    ws = sel.token.str.strip() == ""
    sel = sel[(~ws | sel.is_special).to_numpy() | np.array([(t, int(p)) in early for t, p in zip(sel.tid, sel.pos)], dtype=bool)]
    sel[["tid", "pos"]].to_parquet(out / f"{name}.parquet", index=False)
    (out / f"{name}_tids.json").write_text(json.dumps(sorted(kt)))
    todo = sel[np.array([(t, int(p)) not in done_set for t, p in zip(sel.tid, sel.pos)], dtype=bool)]
    tot[name] = (len(keep_rows), len(sel), len(todo))
    g = todo.groupby("tid").pos.apply(lambda s: sorted(int(x) for x in s))
    tids = list(g.index)
    k = 0
    for c in range(0, len(tids), 400):
        ch = [t for t in tids[c : c + 400]]
        job = {
            "ids": [views[t]["ids"][: max(g[t]) + 1].tolist() for t in ch],
            "pos": [g[t] for t in ch],
            "tids": ch,
            "seed": hash(name) % 10**6 + k,
        }
        if not (jobs / f"{name}_v2__{k:04d}.job.pt").exists():
            torch.save(job, jobs / f"{name}_v2__{k:04d}.job.pt")
        k += 1
    print(name, "episodes", len(keep_rows), "positions", len(sel), "to queue", len(todo), "jobs", k, flush=True)
print("TOTAL", sum(x[1] for x in tot.values()), "to queue", sum(x[2] for x in tot.values()))
