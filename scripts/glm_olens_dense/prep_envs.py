"""prep.py for env-registry runs: the GLM 5.3 envs/<env>/<run> episodes -> one prep dir per run + tiered worker jobs, reading only
positions no read covers yet (coverage.py's exact-prefix test), deduplicated across every run.

    uv run python scripts/glm_olens_dense/prep_envs.py --tok <GLM-5.3 tokenizer dir> --out <prep root> --jobs <job dir>

Policy (Kylie 2026-10-27: every template token, plus punctuation on the simpler tasks), as job tiers the worker runs in order:
  p0  over-eagerness: every template/special token, and punctuation in the head (template header, system, first user message)
  p1  over-eagerness email tasks: punctuation in the later turns
  p2  every other env: template/special tokens and head punctuation; alignment_faking and essay_grading: all punctuation
  p3  hinted_truthfulqa / vulnerable_users: punctuation in the later turns
Donation Bet's later-turn punctuation (4.3M positions, mostly thinking-on reasoning) and the long over-eagerness tasks' are left.

Per prep dir (= collect.py --prep): positions.parquet (new reads), episodes.parquet, dense_prefix_dups.parquet (within the run),
copies.parquet (positions some other read already covers: an earlier upload, or another run of this fill; collect.py copies the
canonical read in, so every episode shows all its reads in one place) and views.pkl.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
import coverage as C
import prep as P
import render as R

PUNCT_TURNS = {"alignment_faking": 2, "essay_grading": 2, "hinted_truthfulqa": 3, "vulnerable_users": 3}


def tier(env: str, task: str, special: bool, head: bool) -> int | None:
    if env == "harness_grid":  # Kylie 2026-10-30: punctuation, delimiters, template tokens; p0 the prompt, p1 the episode
        return 0 if head else 1
    if env == "overeagerness":
        return 0 if special or head else (1 if task.startswith("email") else None)
    return 2 if special or head else PUNCT_TURNS.get(env)


def _read_index(d: str) -> dict[bytes, tuple[str, str, int]]:
    tk = R.tok()
    E = pd.read_parquet(f"{d}/episodes.parquet", columns=["tid", "rendered_text"])
    X = pd.read_parquet(f"{d}/readouts_dense.parquet", columns=["tid", "pos"]).groupby("tid").pos.apply(list).to_dict()
    out = {}
    for tid, text in zip(E.tid, E.rendered_text):
        if tid in X:
            ps = sorted(int(p) for p in X[tid])
            out.update(zip(C.prefix_keys(tk(text, add_special_tokens=False)["input_ids"], ps), ((d, tid, p) for p in ps)))
    return out


def _episode(args):
    run, e, _raw = args
    try:
        r = C._episode(args)
    except Exception as ex:  # noqa: BLE001
        print(f"{run} {e['id']}: {ex!r}", file=sys.stderr)
        return None
    return None if r.get("skip") else r | {"e": {k: e[k] for k in ("id", "arm", "item", "sample")}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--reads", nargs="*", default=C.READS)
    ap.add_argument("--runs", default="data/envs/*/*glm53*")
    ap.add_argument("--positions-per-job", type=int, default=20000)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument(
        "--p1-select", default=None, help="JSON list of 'envs/<env>/<run>/<episode id>': only these get tier-1 (non-prompt) reads"
    )
    a = ap.parse_args()
    p1_select = set(json.loads(Path(a.p1_select).read_text())) if a.p1_select else None
    R.tok(a.tok)
    tk = R.tok()
    out, jobs_dir = Path(a.out), Path(a.jobs)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    dirs = sorted({d for g in a.reads for d in glob.glob(g) if Path(d, "episodes.parquet").exists()})
    owner: dict[bytes, tuple] = {}  # prefix key -> ("old", upload dir, tid, pos) | ("fill", prep name, tid, pos)
    with ProcessPoolExecutor(a.procs, initializer=R.tok, initargs=(a.tok,)) as ex:
        for idx in ex.map(_read_index, dirs):
            owner.update({k: ("old", *v) for k, v in idx.items()})
        print(f"{len(owner):,} read prefixes from {len(dirs)} upload dirs", flush=True)
        runs = list(C.env_runs(a.runs))
        views_by_tier: dict[int, list] = defaultdict(list)
        stats = []
        for run, eps, raw in (x for x in runs if x[1]):
            env, name = run.split("/")[1], run.split("/")[2]
            prep_name = f"{env}__{name}"
            pdir = out / prep_name
            pdir.mkdir(parents=True, exist_ok=True)
            pos_rows, ep_rows, dups, copies = [], [], [], []
            local: dict[bytes, tuple[str, int]] = {}
            for r in ex.map(_episode, [(run, e, raw) for e in eps], chunksize=8):
                if r is None:
                    continue
                e = r["e"]
                tid = "cfi4:" + hashlib.md5(f"{run}/{e['id']}".encode()).hexdigest()[:12]
                task = e["arm"].split("/")[0]
                ids = r["ids"]
                by_tier = defaultdict(list)
                for j, (p, key) in enumerate(zip(r["pos"].tolist(), r["keys"])):
                    if key in local:
                        dups.append(
                            {"tid": tid, "pos": p, "canon_tid": local[key][0], "canon_pos": local[key][1], "canon_in": "readouts_dense"}
                        )
                        continue
                    copied = key in owner
                    if copied:
                        o = owner[key]
                        copies.append({"tid": tid, "pos": p, "src": o[0], "src_dir": o[1], "src_tid": o[2], "src_pos": o[3]})
                    elif (t := tier(env, task, bool(r["special"][j]), bool(r["head"][j]))) is not None and (
                        t != 1 or p1_select is None or f"{run}/{e['id']}" in p1_select
                    ):
                        owner[key] = ("fill", prep_name, tid, p)
                        by_tier[t].append(p)
                    else:
                        continue
                    local[key] = (tid, p)
                    pos_rows.append(
                        {
                            "config": f"glm53_fill_{env}",
                            "run_dir": run,
                            "source_file": "episodes",
                            "row_index": len(ep_rows),
                            "episode_id": e["id"],
                            "env": env,
                            "arm": e["arm"],
                            "item": e["item"],
                            "sample": str(e["sample"]),
                            "positive": False,
                            "positive_source": "none (labels are on the env episode)",
                            "aware": None,
                            "view": "prefix",
                            "msg_index": r["msg_index"][j],
                            "turn": -1,
                            "role": r["role"][j],
                            "pos": p,
                            "token": r["token"][j],
                            "ctx_left40": tk.decode(ids[max(0, p - 40) : p + 1]),
                            "ctx_right40": tk.decode(ids[p + 1 : p + 41]),
                            "next16": tk.decode(ids[p + 1 : p + 17]),
                            "thinking": "off" if raw or "thinkoff" in run else "on",
                            "n_tokens_full": r["n_tokens"],
                            "truncated": r["n_tokens"] > P.CAP,
                            "final_turn_in_view": r["n_tokens"] <= P.CAP,
                            "labels_json": None,
                            "tid": tid,
                            "is_special": bool(r["special"][j]),
                            "copied": copied,
                        }
                    )
                arr = np.asarray(ids[: P.CAP], dtype=np.int32)
                for t, ps in by_tier.items():
                    views_by_tier[t].append({"prep": prep_name, "tid": tid, "ids": arr, "pos": np.asarray(sorted(ps), dtype=np.int32)})
                ep_rows.append(
                    {
                        "tid": tid,
                        "run_dir": run,
                        "source_file": "episodes",
                        "row_index": len(ep_rows),
                        "episode_id": e["id"],
                        "env": env,
                        "arm": e["arm"],
                        "item": e["item"],
                        "n_tokens_full": r["n_tokens"],
                        "truncated": r["n_tokens"] > P.CAP,
                        "rendered_text": r["text"] if r["n_tokens"] <= P.CAP else tk.decode(ids[: P.CAP]),
                        "positions": np.asarray(sorted(r["pos"].tolist()), dtype=np.int32),
                        "tools": "episode",
                    }
                )
            Pd = pd.DataFrame(pos_rows)
            Pd["turn"] = Pd["turn"].astype("int32")
            Pd.to_parquet(pdir / "positions.parquet", index=False)
            pd.DataFrame(ep_rows).to_parquet(pdir / "episodes.parquet", index=False)
            pd.DataFrame(dups, columns=["tid", "pos", "canon_tid", "canon_pos", "canon_in"]).to_parquet(
                pdir / "dense_prefix_dups.parquet", index=False
            )
            pd.DataFrame(copies, columns=["tid", "pos", "src", "src_dir", "src_tid", "src_pos"]).to_parquet(
                pdir / "copies.parquet", index=False
            )
            n_new = int((~Pd.copied).sum()) if len(Pd) else 0
            stats.append({"prep": prep_name, "episodes": len(ep_rows), "new": n_new, "copies": len(copies), "dups": len(dups)})
            print(json.dumps(stats[-1]), flush=True)
    if p1_select is not None and 1 in views_by_tier:  # tier-1 jobs in the selection's order (a priority queue)
        rank = {k: i for i, k in enumerate(json.loads(Path(a.p1_select).read_text()))}
        tid_key = {"cfi4:" + hashlib.md5(k.encode()).hexdigest()[:12]: i for k, i in rank.items()}
        views_by_tier[1].sort(key=lambda v: tid_key.get(v["tid"], len(rank)))
    for t, vs in sorted(views_by_tier.items()):
        k, cur, n = 0, [], 0
        for i, v in enumerate(vs):
            cur.append(v)
            n += len(v["pos"])
            if n >= a.positions_per_job or i == len(vs) - 1:
                job = {
                    "ids": [x["ids"][: int(x["pos"].max()) + 1].tolist() for x in cur],
                    "pos": [x["pos"].tolist() for x in cur],
                    "tids": [x["tid"] for x in cur],
                    "preps": [x["prep"] for x in cur],
                    "seed": (t * 100_000 + k) * 10_000_019,
                }
                torch.save(job, jobs_dir / f"p{t}__{k:04d}.job.pt")
                k, cur, n = k + 1, [], 0
        n_tok = sum(int(v["pos"].max()) + 1 for v in vs)
        print(f"tier p{t}: {k} jobs, {sum(len(v['pos']) for v in vs):,} positions, {n_tok:,} tokens to capture")
    pd.DataFrame(stats).to_csv(out / "prep_stats.csv", index=False)


if __name__ == "__main__":
    main()
