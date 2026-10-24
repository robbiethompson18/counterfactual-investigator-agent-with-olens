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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", type=Path, required=True)
    ap.add_argument("--docent-root", type=Path, action="append", default=[], help="dirs holding runs/<exp>/<run>/docent.json (repeatable)")
    ap.add_argument("--db", type=Path, required=True)
    a = ap.parse_args()
    a.db.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.db.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    docent = docent_map(a.docent_root)
    first = True
    for cfg_dir in sorted(p for p in a.ds.iterdir() if (p / "readouts.parquet").exists()):
        cfg = cfg_dir.name
        r = pd.read_parquet(cfg_dir / "readouts.parquet")
        e = pd.read_parquet(
            cfg_dir / "episodes.parquet",
            columns=["tid", "run_dir", "source_file", "row_index", "rendered_text", "n_tokens_full", "truncated"],
        )
        texts = dict(zip(e.tid, e.rendered_text))
        r = r.sort_values(["tid", "pos"])
        r["char_end"] = -1
        for tid, idx in r.groupby("tid").groups.items():
            if tid in texts:
                r.loc[idx, "char_end"] = locate(texts[tid], r.loc[idx, "ctx_left40"].fillna("").tolist())
        r["positive"] = r.positive.astype(str).str.lower().isin(["true", "1"])
        aw = r.aware.astype(str).str.lower()
        r["aware"] = aw.map({"true": True, "1": True, "false": False, "0": False}).astype("boolean")
        pos = r[POS_COLS + ["char_end"]].copy()
        long = []
        for lens in ("sft", "rl"):
            for s in SAMPLES:
                long.append(pd.DataFrame({"tid": r.tid, "pos": r.pos, "reader": f"glm_{lens}", "sample": s, "text": r[f"{lens}_{s}"]}))
        if (cfg_dir / "qwen_readers.parquet").exists():
            q = pd.read_parquet(cfg_dir / "qwen_readers.parquet")
            for s in SAMPLES:
                long.append(pd.DataFrame({"tid": q.tid, "pos": q.pos, "reader": "qwen_" + q.reader, "sample": s, "text": q[s]}))
        if (cfg_dir / "jlens.parquet").exists():
            j = pd.read_parquet(cfg_dir / "jlens.parquet", columns=["tid", "pos", "jlens_L60_top10", "jlens_L51_top10"])
            for col, name in (("jlens_L60_top10", "jlens_L60"), ("jlens_L51_top10", "jlens_L51")):
                txt = j[col].map(lambda xs: " | ".join(f"{x['token']} {x['prob']:.2f}" for x in xs) if xs is not None else None)
                long.append(pd.DataFrame({"tid": j.tid, "pos": j.pos, "reader": name, "sample": "top10", "text": txt}))
        reads = pd.concat(long, ignore_index=True).dropna(subset=["text"])

        def link(rd: str, sf: str, ri: int) -> str | None:
            local = ALIASES.get(f"{rd}::{sf}", ALIASES.get(rd, rd))
            return docent.get((local, int(ri)))

        links = pd.Series(list(map(link, e.run_dir, e.source_file, e.row_index)), index=e.index, dtype="string")
        eps = e.assign(docent=links, config=cfg)
        for name, df in (("positions", pos), ("reads", reads), ("episodes", eps)):
            con.register("df", df)
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM df" if first else f"INSERT INTO {name} BY NAME SELECT * FROM df")
            con.unregister("df")
        first = False
        located = (pos.char_end >= 0).mean()
        print(f"{cfg}: {len(pos):,} positions ({located:.1%} located in text), {len(reads):,} reads, {len(eps):,} episodes", flush=True)
    con.execute("CREATE INDEX reads_tid ON reads(tid, pos)")
    con.close()
    tmp.replace(a.db)


if __name__ == "__main__":
    main()
