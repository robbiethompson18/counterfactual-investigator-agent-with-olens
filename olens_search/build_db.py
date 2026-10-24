# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.1", "pandas>=2.2", "pyarrow>=17"]
# ///
"""Build the DuckDB file behind the olens regex dashboard from ceselder/glm-5.3-olens-counterfactual-investigator.

Tables:
  positions  one row per read position (tid, pos) with episode identity, labels, token, ±40 context, next 16, and
             `char_end`, the character offset of the read token's end in the episode's rendered text (-1 if not located)
  reads      long format, one row per reader text: (tid, pos, reader, sample, text). Readers: glm_sft / glm_rl
             (greedy, s1..s4), qwen_distilled / qwen_from_scratch (greedy, s1..s4), jlens_L60 / jlens_L51 (top-10 tokens
             joined as "tok | tok | ...", sample = top10)
  episodes   tid -> rendered transcript text, n tokens, truncated flag, Docent URL

    uv run --script olens_search/build_db.py --ds ~/workspace/tmp/ceselder_inventory/ds \
        --docent-root ~/workspace/tmp/ceselder_inventory/ours --db data/olens_search/olens.duckdb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

ALIASES = json.loads((Path(__file__).parent / "docent_aliases.json").read_text())
SAMPLES = ["greedy", "s1", "s2", "s3", "s4"]
POS_COLS = [
    "tid", "pos", "config", "env", "arm", "item", "sample", "run_dir", "row_index", "episode_id", "positive", "positive_source", "aware",
    "view", "msg_index", "turn", "role", "kind", "token", "ctx_left40", "ctx_right40", "next16", "h51_norm", "truncated",
]  # fmt: skip


def docent_map(roots: list[Path]) -> dict[tuple[str, int], str]:
    out = {}
    for root in roots:
        for f in root.glob("runs/*/*/docent.json"):
            d = json.loads(f.read_text())
            rd = str(f.parent.relative_to(root))
            for i, run in enumerate(d["runs"]):
                if run:
                    out[(rd, i)] = f"https://docent.transluce.org/dashboard/{d['collection_id']}/agent_run/{run}"
    return out


def locate(text: str, ctxs: list[str]) -> list[int]:
    """Character offset just past each read token, searching forward (positions come in token order)."""
    out, start = [], 0
    for c in ctxs:
        hit = -1
        for k in (len(c), 160, 80, 40):
            frag = c[-k:]
            if not frag.strip():
                continue
            i = text.find(frag, start)
            if i < 0:
                i = text.find(frag)
            if i >= 0:
                hit = i + len(frag)
                break
        out.append(hit)
        if hit >= 0:
            start = max(0, hit - 1)
    return out


SCHEMA = {
    "positions": """tid VARCHAR, pos BIGINT, source VARCHAR, config VARCHAR, env VARCHAR, arm VARCHAR, item VARCHAR, sample VARCHAR,
        run_dir VARCHAR, source_file VARCHAR, row_index BIGINT, episode_id VARCHAR, positive BOOLEAN,
        positive_source VARCHAR, aware BOOLEAN,
        view VARCHAR, role VARCHAR, kind VARCHAR, token VARCHAR, ctx_left40 VARCHAR, ctx_right40 VARCHAR, next16 VARCHAR, h51_norm DOUBLE,
        labels VARCHAR, char_end BIGINT""",
    "reads": "tid VARCHAR, pos BIGINT, reader VARCHAR, sample VARCHAR, text VARCHAR",
    "episodes": """tid VARCHAR, config VARCHAR, source VARCHAR, run_dir VARCHAR, source_file VARCHAR, row_index BIGINT,
        rendered_text VARCHAR,
        n_tokens_full BIGINT, truncated BOOLEAN, docent VARCHAR""",
}


def as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().map({"true": True, "1": True, "false": False, "0": False}).astype("boolean")


def units(ds: Path) -> list[tuple[str, Path, list[str]]]:
    """(config, dir, readout files) for one downloaded dataset: one subdir per config, or the wild-rollouts single-table layout."""
    if (ds / "olens_wild_rollouts.parquet").exists():
        return [("wild_rollouts", ds, ["olens_wild_rollouts.parquet", "tb21/readouts_dense.parquet"])]
    return [(p.name, p, ["readouts.parquet", "readouts_dense.parquet"]) for p in sorted(ds.iterdir()) if (p / "readouts.parquet").exists()]


def normalise(r: pd.DataFrame, source: str, cfg: str) -> pd.DataFrame:
    """Readout rows from either dataset layout -> the positions schema (missing columns become NULL)."""
    r = r.rename(columns={"left_ctx": "ctx_left40", "file": "source_file"})
    if "family" in r:  # wild-rollouts layout
        r["env"], r["arm"], r["positive"] = r.family, r.task_id, r.passed
    if "env" not in r:
        r["env"] = r.get("task", cfg)
    if "labels_json" in r:
        r["labels"] = r["labels_json"]
    elif "meta_json" in r:
        r["labels"] = r["meta_json"]
    r["source"], r["config"] = source, cfg
    for c in ("positive", "aware", "truncated"):
        r[c] = as_bool(r[c]) if c in r else pd.Series(pd.NA, index=r.index, dtype="boolean")
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ds", type=Path, action="append", required=True, help="HF dataset download dir(s), one config per subdir (repeatable)"
    )
    ap.add_argument("--docent-root", type=Path, action="append", default=[], help="dirs holding runs/<exp>/<run>/docent.json (repeatable)")
    ap.add_argument("--skip", action="append", default=[], help="skip a file, e.g. coding_envs/readouts_dense.parquet (repeatable)")
    ap.add_argument("--db", type=Path, required=True)
    a = ap.parse_args()
    a.db.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.db.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    for name, cols in SCHEMA.items():
        con.execute(f"CREATE TABLE {name} ({cols})")
    docent = docent_map(a.docent_root)

    def insert(name: str, df: pd.DataFrame) -> None:
        con.register("df", df)
        cols = [c.split()[0] for c in SCHEMA[name].replace("\n", " ").split(",")]
        sel = ", ".join(f'"{c}"' if c in df.columns else f"NULL AS {c}" for c in cols)
        con.execute(f"INSERT INTO {name} BY NAME SELECT {sel} FROM df")
        con.unregister("df")

    for ds in a.ds:
        source = ds.name
        for cfg, cfg_dir, files in units(ds):
            files = [f for f in files if (cfg_dir / f).exists() and f"{cfg}/{f}" not in a.skip]
            tx = cfg_dir / "transcripts.parquet"
            e = pd.read_parquet(tx if tx.exists() else cfg_dir / "episodes.parquet").rename(columns={"text": "rendered_text"})
            texts = dict(zip(e.tid, e.rendered_text))
            first_rows = []
            for fname in files:
                r = normalise(pd.read_parquet(cfg_dir / fname), source, cfg).sort_values(["tid", "pos"])
                r["char_end"] = -1
                for tid, idx in r.groupby("tid").groups.items():
                    if tid in texts:
                        r.loc[idx, "char_end"] = locate(texts[tid], r.loc[idx, "ctx_left40"].fillna("").tolist())
                insert("positions", r)
                long = [
                    pd.DataFrame({"tid": r.tid, "pos": r.pos, "reader": f"glm_{lens}", "sample": s, "text": r[f"{lens}_{s}"]})
                    for lens in ("sft", "rl")
                    for s in SAMPLES
                ]
                insert("reads", pd.concat(long, ignore_index=True).dropna(subset=["text"]))
                keep = [c for c in ("tid", "run_dir", "source_file", "row_index") if c in r]
                first_rows.append(r[keep].drop_duplicates("tid"))
                print(f"{source}/{cfg}/{fname}: {len(r):,} positions ({(r.char_end >= 0).mean():.1%} located in text)", flush=True)
            if (cfg_dir / "qwen_readers.parquet").exists():
                q = pd.read_parquet(cfg_dir / "qwen_readers.parquet")
                insert(
                    "reads",
                    pd.concat(
                        [
                            pd.DataFrame({"tid": q.tid, "pos": q.pos, "reader": "qwen_" + q.reader, "sample": s, "text": q[s]})
                            for s in SAMPLES
                        ]
                    ),
                )
            if (cfg_dir / "jlens.parquet").exists():
                j = pd.read_parquet(cfg_dir / "jlens.parquet", columns=["tid", "pos", "jlens_L60_top10", "jlens_L51_top10"])
                for col, name in (("jlens_L60_top10", "jlens_L60"), ("jlens_L51_top10", "jlens_L51")):
                    txt = j[col].map(lambda xs: " | ".join(f"{x['token']} {x['prob']:.2f}" for x in xs) if xs is not None else None)
                    insert(
                        "reads",
                        pd.DataFrame({"tid": j.tid, "pos": j.pos, "reader": name, "sample": "top10", "text": txt}).dropna(subset=["text"]),
                    )
            ids = pd.concat(first_rows).drop_duplicates("tid")
            eps = e[["tid", "rendered_text", "n_tokens_full", "truncated"]].merge(ids, on="tid", how="left")
            for c in ("run_dir", "source_file"):
                if c not in eps:
                    eps[c] = None
            if "row_index" not in eps:
                eps["row_index"] = -1

            def link(rd: object, sf: object, ri: object) -> str | None:
                if not isinstance(rd, str) or pd.isna(ri):
                    return None
                local = ALIASES.get(f"{rd}::{sf}", ALIASES.get(rd, rd))
                return docent.get((local, int(ri)))

            eps["docent"] = pd.Series(list(map(link, eps.run_dir, eps.source_file, eps.row_index)), index=eps.index, dtype="string")
            eps["config"], eps["source"] = cfg, source
            eps["truncated"] = as_bool(eps.truncated)
            insert("episodes", eps)
            print(f"  {cfg}: {len(eps):,} transcripts", flush=True)
    con.execute("CREATE INDEX reads_tid ON reads(tid, pos)")
    con.close()
    tmp.replace(a.db)


if __name__ == "__main__":
    main()
