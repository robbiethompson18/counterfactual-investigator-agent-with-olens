"""Join worker outputs with prep positions into Celeste-schema readouts_dense.parquet (+ episodes, dups, empty readouts) for one run dir."""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

S, I32, F, B = pa.string(), pa.int32(), pa.float32(), pa.bool_()
COLS = (
    [
        ("config", S),
        ("run_dir", S),
        ("source_file", S),
        ("row_index", I32),
        ("episode_id", S),
        ("env", S),
        ("arm", S),
        ("item", S),
        ("sample", S),
        ("positive", B),
        ("positive_source", S),
        ("aware", B),
        ("view", S),
        ("msg_index", I32),
        ("turn", I32),
        ("role", S),
        ("kind", S),
        ("pos", I32),
        ("token", S),
        ("ctx_left40", S),
        ("ctx_right40", S),
        ("next16", S),
    ]
    + [(f"sft_{x}", S) for x in ("greedy", "s1", "s2", "s3", "s4")]
    + [(f"rl_{x}", S) for x in ("greedy", "s1", "s2", "s3", "s4")]
    + [
        ("h51_norm", F),
        ("thinking", S),
        ("n_tokens_full", I32),
        ("truncated", B),
        ("final_turn_in_view", B),
        ("labels_json", S),
        ("tid", S),
        ("dense", B),
        ("prefix_cache", B),
        ("lens_cache_salt", S),
        ("part", S),
    ]
    + [(f"fve_{lens}_{x}", F) for lens in ("sft", "rl") for x in ("greedy", "s1", "s2", "s3", "s4")]
)
SCHEMA = pa.schema(COLS)
READ = ["rl_greedy", "rl_s1", "rl_s2", "rl_s3", "rl_s4"]


def shard_rows(job: Path, P: pd.DataFrame) -> pd.DataFrame:
    j, o = torch.load(job), torch.load(str(job).replace(".job.pt", ".out.pt"))
    recs = []
    for tid, pos, norms, reads in zip(j["tids"], j["pos"], o["h51_norm"], o["rl"]):
        for p, nrm, r in zip(pos, norms, reads):
            recs.append((tid, int(p), float(nrm), *r))
    R = pd.DataFrame(recs, columns=["tid", "pos", "h51_norm", *READ])
    return P.merge(R, on=["tid", "pos"], how="inner", validate="one_to_one")


def to_table(df: pd.DataFrame, kind: str, dense: bool) -> pa.Table:
    df = df.copy()
    df["kind"] = kind
    # worker --prefix-cache: steered KV salted per vector, greedy + samples of one position share it (Celeste's lens_cache_salt=lens-sha256)
    df["dense"], df["prefix_cache"], df["lens_cache_salt"] = dense, True, "lens-sha256"
    for c, t in COLS:
        if c not in df:
            df[c] = None
    return pa.Table.from_pandas(df[[c for c, _ in COLS]], schema=SCHEMA, preserve_index=False)


def add_copies(prep: Path, out: Path, fill_root: Path):
    """Positions another read already covers (copies.parquet: an earlier upload's readouts_dense, or another fill run's): copy the
    canonical row's reads under this run's tid/pos, kind=copied, so the episode shows every read in one upload."""
    C = pd.read_parquet(prep / "copies.parquet")
    if not len(C):
        return
    src = []
    for (kind, d), g in C.groupby(["src", "src_dir"]):
        f = Path(d) / "readouts_dense.parquet" if kind == "old" else fill_root / d / "readouts_dense.parquet"
        S = pd.read_parquet(f, columns=["tid", "pos", "h51_norm", *[c for c, _ in COLS if c.startswith(("rl_", "sft_", "fve_"))]])
        S = S.drop_duplicates(["tid", "pos"]).rename(columns={"tid": "src_tid", "pos": "src_pos"})
        src.append(g.merge(S, on=["src_tid", "src_pos"], how="inner", validate="one_to_one"))
    S = pd.concat(src).drop(columns=["src", "src_dir", "src_tid", "src_pos"])
    P = pd.read_parquet(prep / "positions.parquet").drop(columns=["is_special", "copied"])
    rows = P.merge(S, on=["tid", "pos"], how="inner", validate="one_to_one")
    old = pq.read_table(out / "readouts_dense.parquet")
    old = old.filter(pc.field("kind") != "copied") if old.num_rows else old
    pq.write_table(pa.concat_tables([old, to_table(rows, "copied", True)]), out / "readouts_dense.parquet", compression="zstd")
    print(f"{prep.name}: {len(rows):,} copied rows of {len(C):,} ({old.num_rows:,} read here)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--jobs", required=True, help="dir holding <run>__NNNN.job.pt / .out.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", default="dense", help="`kind` value (spread64 for a per-episode subset)")
    ap.add_argument(
        "--subset", action="store_true", help="per-episode subset: dense=False; prefix dups and episodes.positions limited to what was read"
    )
    ap.add_argument("--keep", default=None, help="parquet of (tid, pos) to keep (drops extra reads of an earlier, wider job)")
    ap.add_argument("--tids", default=None, help="JSON list of the sampled episode views: episodes + prefix dups restricted to them")
    ap.add_argument("--jobs-glob", default=None, help="job files to scan when jobs mix runs (prep_envs.py: p<tier>__NNNN.job.pt)")
    ap.add_argument(
        "--copies-from", default=None, help="second pass (prep_envs.py): append copies.parquet's reads; fill sources under this root"
    )
    a = ap.parse_args()
    if a.copies_from:
        return add_copies(Path(a.prep), Path(a.out), Path(a.copies_from))
    prep, out = Path(a.prep), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    P = pd.read_parquet(prep / "positions.parquet").drop(columns=["is_special", "copied"], errors="ignore")
    jobs = sorted(Path(p) for p in glob.glob(a.jobs_glob or f"{a.jobs}/{prep.name}*__*.job.pt"))
    done = [j for j in jobs if Path(str(j).replace(".job.pt", ".out.pt")).exists()]
    w = pq.ParquetWriter(out / "readouts_dense.parquet", SCHEMA, compression="zstd")
    keep = pd.read_parquet(a.keep).drop_duplicates() if a.keep else None
    tids = set(json.loads(Path(a.tids).read_text())) if a.tids else None
    n, read_pos = 0, []
    for j in done:
        kind = "head" if "head__" in j.name else a.kind  # --head jobs: every position through the first user message
        rows = shard_rows(j, P)
        if a.keep:
            rows = rows.merge(keep, on=["tid", "pos"])
        t = to_table(rows, kind, not a.subset)
        read_pos.append(t.select(["tid", "pos"]).to_pandas())
        w.write_table(t, row_group_size=50_000)
        n += t.num_rows
    w.close()
    eps = pd.read_parquet(prep / "episodes.parquet")
    dups = pd.read_parquet(prep / "dense_prefix_dups.parquet")
    if tids is not None:
        eps, dups = eps[eps.tid.isin(tids)], dups[dups.tid.isin(tids)]
    if a.subset:  # only the positions actually read, plus the exact-prefix duplicates of those (read once, shown in every episode)
        rp = pd.concat(read_pos)
        dups = dups.merge(rp.rename(columns={"tid": "canon_tid", "pos": "canon_pos"}), on=["canon_tid", "canon_pos"])
        allp = pd.concat([rp, dups[["tid", "pos"]]]).groupby("tid").pos.apply(lambda x: np.sort(x.to_numpy().astype("int32")))
        eps["positions"] = eps.tid.map(allp).map(lambda x: x if isinstance(x, np.ndarray) else np.zeros(0, "int32"))
    dups.to_parquet(out / "dense_prefix_dups.parquet", index=False)
    eps.to_parquet(out / "episodes.parquet", index=False)
    pq.write_table(
        SCHEMA.empty_table().drop_columns(["dense", "prefix_cache", "lens_cache_salt"] + [c for c, _ in COLS if c.startswith("fve_")]),
        out / "readouts.parquet",
    )  # the importer looks for a readouts.parquet per config dir; every read here is dense
    print(f"{prep.name}: {len(done)}/{len(jobs)} jobs, {n:,} rows of {len(P):,} positions")


if __name__ == "__main__":
    main()
