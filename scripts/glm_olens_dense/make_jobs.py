"""Split prep views into worker jobs of ~N positions: <out>/<run>__NNNN.job.pt = {"ids", "pos", "tids", "seed"}."""

import argparse
import pickle
from pathlib import Path

import torch


def spread(prep: Path, views: list[dict], k: int) -> list[dict]:
    """K positions per episode, evenly spaced (by rank) over its candidates in full-episode token order. A late view's positions
    map to full-episode indices by aligning its end with the episode's end (the kept tail is verbatim); only those past the prefix
    view's cut (>= 32,768) are candidates, so nothing is counted twice."""
    import numpy as np
    import pandas as pd

    P = pd.read_parquet(prep / "positions.parquet", columns=["tid", "pos", "row_index", "view", "n_tokens_full"])
    P = P[P.tid.isin({v["tid"] for v in views})]
    full = P[P.view == "prefix"].groupby("row_index").n_tokens_full.first()
    lv = P.view == "late"
    P["gpos"] = P.pos
    P.loc[lv, "gpos"] = P.pos[lv] + P.row_index[lv].map(full) - P.n_tokens_full[lv]
    P = P[~lv | (P.gpos >= 32768)].sort_values(["row_index", "gpos"])
    pick = []
    for _, g in P.groupby("row_index"):
        idx = np.unique(np.round(np.linspace(0, len(g) - 1, min(k, len(g)))).astype(int))
        pick.append(g.iloc[idx])
    S = pd.concat(pick)
    keep = S.groupby("tid").pos.apply(lambda s: np.sort(s.to_numpy().astype(np.int32))).to_dict()
    print(f"spread {k}/episode: {len(S):,} positions over {S.row_index.nunique()} episodes")
    return [dict(v, pos=keep.get(v["tid"], np.zeros(0, np.int32))) for v in views]


def head(prep: Path, views: list[dict], exclude: str) -> list[dict]:
    """positions in the template header (tool spec), the system prompt and the first user message. These runs' episodes all open
    [system, user], so that is msg_index <= 1 (-1 = template header); positions are already exact-prefix deduplicated by prep."""
    import glob

    import numpy as np
    import pandas as pd

    P = pd.read_parquet(prep / "positions.parquet", columns=["tid", "pos", "msg_index"])
    P = P[P.msg_index <= 1]
    done = set()
    for f in glob.glob(exclude) if exclude else []:
        j = torch.load(f)
        done |= {(t, int(x)) for t, ps in zip(j["tids"], j["pos"]) for x in ps}
    P = P[[(t, int(x)) not in done for t, x in zip(P.tid, P.pos)]]
    keep = P.groupby("tid").pos.apply(lambda s: np.sort(s.to_numpy().astype(np.int32))).to_dict()
    print(f"head: {len(P):,} unique positions over {P.tid.nunique()} views ({len(done):,} already read excluded)")
    return [dict(v, pos=keep.get(v["tid"], np.zeros(0, np.int32))) for v in views]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--positions-per-job", type=int, default=50000)
    ap.add_argument(
        "--per-episode",
        type=int,
        default=0,
        help="read only K positions per episode, evenly spaced over its candidates across the full token range (late view = tail)",
    )
    ap.add_argument(
        "--only",
        choices=["all", "newline_special", "rest"],
        default="all",
        help="newline_special = template/special tokens + tokens containing a newline; rest = the other punctuation",
    )
    ap.add_argument("--head", action="store_true", help="every unique position through the first user message")
    ap.add_argument("--exclude-jobs", default="", help="glob of job files whose positions are already read (left out)")
    ap.add_argument("--skip-arms", default="", help="comma list of arms to leave out (e.g. superseded cells)")
    ap.add_argument("--tag", default="", help="job-name suffix after the run name (keeps a second pass's jobs apart)")
    a = ap.parse_args()
    prep, out = Path(a.prep), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(prep / "views.pkl", "rb") as f:
        views = pickle.load(f)
    if a.only != "all":
        import numpy as np
        import pandas as pd

        P = pd.read_parquet(prep / "positions.parquet", columns=["tid", "pos", "token", "is_special"])
        sel = P.is_special | P.token.str.contains("\n", regex=False)
        P = P[sel if a.only == "newline_special" else ~sel]
        keep = P.groupby("tid").pos.apply(lambda s: np.sort(s.to_numpy().astype(np.int32))).to_dict()
        views = [dict(v, pos=keep.get(v["tid"], np.zeros(0, np.int32))) for v in views]
    if a.skip_arms:
        import pandas as pd

        E = pd.read_parquet(prep / "episodes.parquet", columns=["tid", "arm"])
        drop = set(E[E.arm.isin(a.skip_arms.split(","))].tid)
        views = [v for v in views if v["tid"] not in drop]
        print(f"skip-arms: {len(drop)} views left out")
    if a.per_episode:
        views = spread(prep, views, a.per_episode)
    if a.head:
        views = head(prep, views, a.exclude_jobs)
    views = [v for v in views if len(v["pos"])]
    k, cur, n = 0, [], 0

    def flush():
        nonlocal k, cur, n
        if cur:
            job = {
                "ids": [v["ids"][: int(v["pos"].max()) + 1].tolist() for v in cur],
                "pos": [v["pos"].tolist() for v in cur],
                "tids": [v["tid"] for v in cur],
                "seed": k * 10_000_019,
            }
            torch.save(job, out / f"{prep.name}{a.tag}__{k:04d}.job.pt")
            k += 1
        cur, n = [], 0

    for v in views:
        cur.append(v)
        n += len(v["pos"])
        if n >= a.positions_per_job:
            flush()
    flush()
    print(f"{k} jobs, {sum(len(v['pos']) for v in views):,} positions")


if __name__ == "__main__":
    main()
