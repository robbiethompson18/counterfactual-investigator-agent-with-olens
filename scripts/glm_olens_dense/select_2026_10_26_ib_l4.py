"""IB L4 steve_repo + L4_assistant_formal top-up: head (every unique punct/template position through the first user message)
+ 64 evenly spaced per episode over its candidates; whitespace-only tokens dropped."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

name = "l4_steve"
prep = Path("prep") / name
P = pd.read_parquet(
    prep / "positions.parquet", columns=["tid", "pos", "row_index", "view", "n_tokens_full", "msg_index", "token", "is_special"]
)
P = P[((P.token.str.strip() != "") | P.is_special).to_numpy()]
head = P[P.msg_index <= 1]
full = P[P.view == "prefix"].groupby("row_index").n_tokens_full.first()
lv = P.view == "late"
P = P.assign(gpos=P.pos)
P.loc[lv, "gpos"] = P.pos[lv] + P.row_index[lv].map(full) - P.n_tokens_full[lv]
C = P[~lv | (P.gpos >= 32768)].sort_values(["row_index", "gpos"])
pick = [g.iloc[np.unique(np.round(np.linspace(0, len(g) - 1, min(64, len(g)))).astype(int))] for _, g in C.groupby("row_index")]
S = pd.concat(pick + [head]).drop_duplicates(["tid", "pos"])
kind = pd.Series("spread64", index=pd.MultiIndex.from_frame(S[["tid", "pos"]]))
kind[pd.MultiIndex.from_frame(head[["tid", "pos"]])] = "head"
S[["tid", "pos"]].assign(kind=kind.values).to_parquet(f"sel_{name}.parquet", index=False)
views = {v["tid"]: v for v in pickle.loads((prep / "views.pkl").read_bytes())}
g = S.groupby("tid").pos.apply(lambda s: sorted(int(x) for x in s))
Path("jobs_l4").mkdir(exist_ok=True)
tids, k, cur, n = list(g.index), 0, [], 0
for t in tids + [None]:
    if t is not None:
        cur.append(t)
        n += len(g[t])
    if cur and (t is None or n >= 15000):
        torch.save(
            {"ids": [views[x]["ids"][: max(g[x]) + 1].tolist() for x in cur], "pos": [g[x] for x in cur], "tids": cur, "seed": 11 + k},
            f"jobs_l4/{name}__{k:04d}.job.pt",
        )
        k += 1
        cur, n = [], 0
print(f"episodes {C.row_index.nunique()} spread {len(S) - len(head)} head {len(head)} total {len(S)} jobs {k}")
