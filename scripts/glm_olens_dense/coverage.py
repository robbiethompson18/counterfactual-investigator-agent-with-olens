"""Which GLM 5.3 env episodes have O-lens reads, and on which tokens.

For every episode of every `envs/<env>/<run>` whose target is GLM 5.3, lists prep.py's candidate positions (punctuation /
newline + template/special tokens, index >= 2, < 32,768) and checks each against every read already made, by exact token
prefix: a read covers a position when its transcript's tokens up to and including the read position are identical. So
reads made on the pre-registry source rows (email_ambiguous, the 2026-10-26 over-eagerness rows, AF reworded) count
wherever the env episode renders to the same tokens, whatever ids they were filed under.

    uv run python scripts/glm_olens_dense/coverage.py --tok <GLM-5.3 tokenizer dir> --out data/runs/olens_reads/coverage \
        [--reads 'data/runs/olens_reads/*glm53*/*' ...] [--runs 'data/envs/*/*glm53*']

Writes `<out>/coverage.parquet` (one row per episode: candidates and covered counts by kind x section), `<out>/missing.parquet`
(each uncovered candidate, exact-prefix deduplicated across all episodes: the fill list) and prints a per-run table.
kind = special (id >= 154820: chat template, think / tool-call tags) | punct; section = head (template header, system prompt,
first user message) | turns (everything after).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import prep as P
import render as R

READS = [
    "data/runs/olens_reads/*glm53*/*",
    "data/runs/alignment_faking/2026-10-27_olens-reworded/glm53/*",
]


def prefix_keys(ids, positions) -> list[bytes]:
    """blake2b of ids[: p + 1] for each p (sorted), incrementally, as prep.py keys its dedup."""
    arr = np.asarray(ids, dtype=np.int32)
    h, last, out = hashlib.blake2b(digest_size=16), 0, []
    for p in sorted(positions):
        h.update(arr[last : p + 1].tobytes())
        last = p + 1
        out.append(h.copy().digest())
    return out


def _read_keys(d: str) -> set[bytes]:
    tk = R.tok()
    E = pd.read_parquet(f"{d}/episodes.parquet", columns=["tid", "rendered_text"])
    pos = []
    for f in ("readouts_dense.parquet", "readouts.parquet"):
        if Path(d, f).exists():
            x = pd.read_parquet(f"{d}/{f}", columns=["tid", "pos"])
            pos.append(x)
    X = pd.concat(pos).groupby("tid").pos.apply(list).to_dict()
    keys = set()
    for tid, text in zip(E.tid, E.rendered_text):
        if tid in X:
            keys.update(prefix_keys(tk(text, add_special_tokens=False)["input_ids"], [int(p) for p in X[tid]]))
    return keys


def render_episode(e: dict, raw: bool) -> str:
    return P.raw_render(e["messages"]) if raw else R.render(e["messages"], e.get("tools") or None)


def _episode(args):
    run, e, raw = args
    tk = R.tok()
    msgs, tools = e["messages"], e.get("tools") or None
    try:
        text = render_episode(e, raw)
    except AssertionError:  # raw render: the episode doesn't end in a plain reply (errored / empty)
        return {"run": run, "episode": e["id"], "arm": e["arm"], "skip": True}
    ids, n_full, prs = P.view_rows(tk, text, msgs, tools, lambda i: i, 0, {}, raw)
    fu = next((i for i, m in enumerate(msgs) if m["role"] == "user"), 0)
    keys = prefix_keys(ids[: P.CAP], [d["pos"] for d in prs])
    return {
        "run": run,
        "episode": e["id"],
        "arm": e["arm"],
        "n_tokens": n_full,
        "pos": np.asarray([d["pos"] for d in prs], dtype=np.int32),
        "special": np.asarray([d["special"] for d in prs], dtype=bool),
        "head": np.asarray([d["msg_index"] <= fu for d in prs], dtype=bool),
        "newline": np.asarray(["\n" in d["token"] for d in prs], dtype=bool),
        "role": [d["role"] for d in prs],
        "token": [d["token"] for d in prs],
        "msg_index": [d["msg_index"] for d in prs],
        "keys": keys,
        "ids": ids,
        "text": text,
    }


def env_runs(pattern: str):
    for r in sorted(glob.glob(pattern)):
        meta = json.loads(Path(r, "meta.json").read_text())
        if "glm-5.3" not in meta.get("target", {}).get("model", ""):
            continue
        raw = meta["target"].get("backend") == "openrouter-raw"
        eps = [json.loads(x) for f in sorted(glob.glob(f"{r}/episodes/*.jsonl")) for x in Path(f).read_text().splitlines()]
        yield str(Path(r).relative_to("data")), eps, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reads", nargs="*", default=READS)
    ap.add_argument("--runs", default="data/envs/*/*glm53*")
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    R.tok(a.tok)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dirs = sorted({d for g in a.reads for d in glob.glob(g) if Path(d, "episodes.parquet").exists()})
    with ProcessPoolExecutor(a.procs, initializer=R.tok, initargs=(a.tok,)) as ex:
        read = set().union(*ex.map(_read_keys, dirs))
        print(f"{len(read):,} read prefixes from {len(dirs)} upload dirs", flush=True)
        jobs = [(run, e, raw) for run, eps, raw in env_runs(a.runs) for e in eps]
        print(f"{len(jobs):,} episodes", flush=True)
        rows, missing, seen = [], [], set()
        for i, r in enumerate(ex.map(_episode, jobs, chunksize=16)):
            if r.get("skip"):
                rows.append({k: r[k] for k in ("run", "episode", "arm")} | {"unrendered": True})
                continue
            cov = np.asarray([k in read for k in r["keys"]], dtype=bool)
            row = {k: r[k] for k in ("run", "episode", "arm", "n_tokens")}
            for kind, km in (("special", r["special"]), ("punct", ~r["special"])):
                for sec, sm in (("head", r["head"]), ("turns", ~r["head"])):
                    m = km & sm
                    row[f"{kind}_{sec}"] = int(m.sum())
                    row[f"{kind}_{sec}_read"] = int((m & cov).sum())
            rows.append(row)
            for j in np.flatnonzero(~cov):
                k = r["keys"][j]
                if k in seen:
                    continue
                seen.add(k)
                missing.append(
                    {
                        "run": r["run"],
                        "episode": r["episode"],
                        "pos": int(r["pos"][j]),
                        "special": bool(r["special"][j]),
                        "head": bool(r["head"][j]),
                        "newline": bool(r["newline"][j]),
                        "role": r["role"][j],
                    }
                )
            if i % 2000 == 0:
                print(f"{i:,}/{len(jobs):,}", flush=True)
    C = pd.DataFrame(rows)
    C.to_parquet(out / "coverage.parquet", index=False)
    pd.DataFrame(missing).to_parquet(out / "missing.parquet", index=False)
    print(table(C, pd.DataFrame(missing)))


def table(C: pd.DataFrame, M: pd.DataFrame) -> str:
    cols = [f"{k}_{s}" for k in ("special", "punct") for s in ("head", "turns")]
    g = C.assign(any_read=C[[c + "_read" for c in cols]].sum(axis=1) > 0).groupby("run")
    C = C.assign(unrendered=C.get("unrendered", False)).fillna({"unrendered": False})
    g = C.assign(any_read=C[[c + "_read" for c in cols]].sum(axis=1) > 0).groupby("run")
    T = g.agg(
        episodes=("episode", "size"),
        unrendered=("unrendered", "sum"),
        arms=("arm", "nunique"),
        with_reads=("any_read", "sum"),
        tokens=("n_tokens", "median"),
    )
    for c in cols:
        T[c] = g[c + "_read"].sum().astype(str) + "/" + g[c].sum().astype(str)
    if len(M):
        T["missing_unique"] = M.groupby("run").size()
        T["missing_unique_special"] = M[M.special].groupby("run").size()
    return T.fillna(0).to_markdown()


if __name__ == "__main__":
    main()
