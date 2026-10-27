# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb>=1.1", "fastapi>=0.115", "uvicorn>=0.30"]
# ///
"""Regex search over every O-lens / Qwen-reader / J-lens readout of our GLM-5.3 transcripts, with transcripts inline.

    uv run --script olens_search/server.py --db data/olens_search/olens.duckdb --host 100.86.34.70 --host fd7a:... --port 8821

Each --host gets its own listening socket (bind to the tailnet addresses only). See olens_search/CLAUDE.md.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import socket
import threading
from pathlib import Path
from typing import Annotated

import duckdb
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

STATIC = Path(__file__).parent / "static"
app = FastAPI()
DB: duckdb.DuckDBPyConnection
LOCK = threading.Lock()


def q(sql: str, params: list | None = None) -> list[dict]:
    with LOCK:
        cur = DB.cursor()
        res = cur.execute(sql, params or [])
        cols = [d[0] for d in res.description]
        return [dict(zip(cols, row)) for row in res.fetchall()]


class Filters:
    """Query-string filters shared by /api/search and /api/csv."""

    def __init__(
        self, pattern: str, case: str, novel: str, readers: str, configs: str, env: str, arm: str, kind: str, role: str, label: str
    ):
        if not pattern:
            raise HTTPException(400, "empty pattern")
        try:
            re.compile(pattern)
        except re.error as err:
            raise HTTPException(400, f"bad regex: {err}") from err
        self.rx = pattern if case == "1" else f"(?i){pattern}"
        self.novel, self.readers = novel, [v for v in readers.split(",") if v]
        self.slice_conds, self.slice_params = [], []  # over positions p only
        for col, val in (("p.config", configs), ("p.env", env), ("p.arm", arm), ("p.role", role)):
            vals = [v for v in val.split(",") if v]
            if vals:
                self.slice_conds.append(f"{col} IN ({','.join('?' * len(vals))})")
                self.slice_params += vals
        kinds = [v for v in kind.split(",") if v]
        if kinds:  # kinds are '+'-joined
            self.slice_conds.append("(" + " OR ".join("list_contains(string_split(p.kind, '+'), ?)" for _ in kinds) + ")")
            self.slice_params += kinds
        lab = {"positive": "p.positive", "negative": "NOT p.positive", "aware": "p.aware", "unaware": "p.aware = false"}.get(label)
        if lab:
            self.slice_conds.append(lab)

    def slice_sql(self) -> tuple[str, list]:
        return " AND ".join(self.slice_conds) or "TRUE", list(self.slice_params)

    def hit_sql(self) -> tuple[str, list]:
        """Predicate over reads r JOIN positions p JOIN episodes e."""
        conds, params = ["regexp_matches(r.text, ?)"], [self.rx]
        if self.novel in ("ctx", "transcript"):
            conds.append(
                "NOT regexp_matches(coalesce(p.ctx_left40,'') || ' ' || coalesce(p.ctx_right40,'') || ' ' || coalesce(p.next16,''), ?)"
            )
            params.append(self.rx)
        if self.novel == "transcript":
            conds.append("NOT regexp_matches(coalesce(e.rendered_text, ''), ?)")
            params.append(self.rx)
        if self.readers:
            conds.append(f"r.reader IN ({','.join('?' * len(self.readers))})")
            params += self.readers
        sw, sp = self.slice_sql()
        return " AND ".join(conds) + " AND " + sw, params + sp


def filters(
    q_: str = Query("", alias="q"), case: str = "0", novel: str = "none", readers: str = "", configs: str = "", env: str = "",
    arm: str = "",
    kind: str = "", role: str = "", label: str = "",
) -> Filters:  # fmt: skip
    return Filters(q_, case, novel, readers, configs, env, arm, kind, role, label)


FROM = "reads r JOIN positions p USING (tid, pos) LEFT JOIN episodes e ON e.tid = p.tid"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/meta")
def meta() -> dict:
    return {
        "configs": q("SELECT config, any_value(source) source, count(*) n, count(DISTINCT tid) eps FROM positions GROUP BY 1 ORDER BY 1"),
        "envs": q("SELECT config, env, arm, count(*) n FROM positions GROUP BY ALL ORDER BY ALL"),
        "readers": q("SELECT reader, count(*) n FROM reads GROUP BY 1 ORDER BY 1"),
        "kinds": q("SELECT DISTINCT unnest(string_split(kind, '+')) kind FROM positions ORDER BY 1"),
        "roles": q("SELECT DISTINCT role FROM positions ORDER BY 1"),
    }


@app.get("/api/search")
def search(f: Annotated[Filters, Depends(filters)], page: int = 0, per: int = 50) -> JSONResponse:
    hw, hp = f.hit_sql()
    sw, sp = f.slice_sql()
    with LOCK:
        cur = DB.cursor()
        cur.execute(f"CREATE OR REPLACE TEMP TABLE hit_reads AS SELECT r.tid, r.pos, r.reader, r.sample, r.text FROM {FROM} WHERE {hw}", hp)

        def fetch(sql: str, params: list | None = None) -> list[dict]:
            res = cur.execute(sql, params or [])
            cols = [d[0] for d in res.description]
            return [dict(zip(cols, row)) for row in res.fetchall()]

        tot = fetch("SELECT count(*) reads, count(DISTINCT (tid, pos)) positions, count(DISTINCT tid) episodes FROM hit_reads")[0]
        # per (config, env, arm): episodes in the filtered slice, how many have a hit, and label rates with vs overall
        by_slice = fetch(
            f"""WITH hp AS (SELECT DISTINCT tid, pos FROM hit_reads),
            s AS (SELECT p.config, p.env, p.arm, p.episode_id, p.positive, p.aware, (hp.tid IS NOT NULL) hit
                  FROM positions p LEFT JOIN hp USING (tid, pos) WHERE {sw}),
            ep AS (SELECT config, env, arm, episode_id, any_value(positive) positive, any_value(aware) aware, bool_or(hit) hit
                   FROM s GROUP BY ALL)
            SELECT config, env, arm, count(*) episodes, sum(hit::int) hit_eps,
                   avg(positive::int) pos_rate, avg(CASE WHEN hit THEN positive::int END) pos_rate_hit,
                   avg(aware::int) aware_rate, avg(CASE WHEN hit THEN aware::int END) aware_rate_hit
            FROM ep GROUP BY ALL HAVING sum(hit::int) > 0 ORDER BY hit_eps DESC, config, env, arm""",
            sp,
        )
        rows = fetch(
            """WITH hp AS (SELECT tid, pos, list({'reader': reader, 'sample': sample, 'text': text} ORDER BY reader, sample) reads
                         FROM hit_reads GROUP BY tid, pos ORDER BY tid, pos LIMIT ? OFFSET ?)
            SELECT p.tid, p.pos, p.config, p.env, p.arm, p.run_dir, p.row_index, p.kind, p.role, p.token, p.positive, p.aware, p.view,
                   right(p.ctx_left40, 260) ctx, p.next16, e.docent, hp.reads
            FROM hp JOIN positions p USING (tid, pos) LEFT JOIN episodes e ON e.tid = p.tid ORDER BY p.tid, p.pos""",
            [per, page * per],
        )
    return JSONResponse({"total": tot, "by_slice": by_slice, "rows": rows, "rx": f.rx})


@app.get("/api/csv")
def export(f: Annotated[Filters, Depends(filters)]) -> StreamingResponse:
    hw, hp = f.hit_sql()
    rows = q(
        f"""SELECT p.config, p.env, p.arm, p.run_dir, p.row_index, p.tid, p.pos, p.kind, p.role, p.token, p.positive, p.aware,
            r.reader, r.sample, r.text, p.ctx_left40, p.next16, e.docent
            FROM {FROM} WHERE {hw} ORDER BY p.config, p.tid, p.pos LIMIT 500000""",
        hp,
    )
    buf = io.StringIO()
    if rows:
        wr = csv.DictWriter(buf, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    headers = {"Content-Disposition": "attachment; filename=olens_search.csv"}
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers=headers)


@app.get("/api/episode")
def episode(tid: str) -> dict:
    e = q("SELECT * FROM episodes WHERE tid = ?", [tid])
    if not e:
        raise HTTPException(404, "no such episode")
    pos = q(
        """SELECT p.pos, p.char_end, p.kind, p.role, p.token, p.next16, p.positive, p.aware, p.env, p.arm, p.config, p.view, p.row_index,
                  list({'reader': r.reader, 'sample': r.sample, 'text': r.text} ORDER BY r.reader, r.sample) reads
           FROM positions p JOIN reads r USING (tid, pos) WHERE p.tid = ? GROUP BY ALL ORDER BY p.pos""",
        [tid],
    )
    return {"episode": e[0], "positions": pos}


def main() -> None:
    global DB
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--host", action="append", required=True)
    ap.add_argument("--port", type=int, default=8821)
    a = ap.parse_args()
    DB = duckdb.connect(str(a.db), read_only=True)
    DB.execute("SET threads TO 4")
    socks = []
    for h in a.host:
        fam = socket.AF_INET6 if ":" in h else socket.AF_INET
        s = socket.socket(fam, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if fam == socket.AF_INET6:
            s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        s.bind((h, a.port))
        socks.append(s)
    uvicorn.Server(uvicorn.Config(app, log_level="warning")).run(sockets=socks)


if __name__ == "__main__":
    main()
