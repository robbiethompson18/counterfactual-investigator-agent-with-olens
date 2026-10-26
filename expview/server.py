# /// script
# requires-python = ">=3.12"
# dependencies = ["duckdb>=1.1", "pyarrow>=17", "fastapi>=0.115", "uvicorn>=0.30"]
# ///
"""expview server: every run under the roots (expview.roots()) in one UI, with regex search over all metamodel reads.

    uv run --script expview/server.py --host 100.86.34.70 --host fd7a:115c:a1e0::e934:2247 --port 8830

Each --host gets its own listening socket (bind to the tailnet addresses only; expview/run.sh does this on bramble).
The index (default data/expview/index.duckdb) is updated at start and then every --rescan seconds for runs whose files
changed; POST /api/reindex?d=<id> forces one run. See expview/README.md.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

sys.path.insert(0, str(Path(__file__).parent))
from index import Index

from expview import REPO, load_tokens, merge_overlay, points_at, resolve_env

PAGE = Path(__file__).parent / "static/index.html"
app = FastAPI()
IX: Index


def run_dir(d: str) -> Path:
    p = IX.paths.get(d)
    if p is None:
        raise HTTPException(404, f"no dataset {d!r} (indexed: {len(IX.paths)}; it may still be indexing)")
    return p


@app.get("/")
def index() -> HTMLResponse:
    page = PAGE.read_text().replace("{{TITLE}}", "expview").replace("{{DATA}}", "")
    return HTMLResponse(page)


@app.get("/api/status")
def status() -> dict:
    return IX.status


@app.get("/api/datasets")
def datasets() -> list[dict]:
    """Every dataset, with how runs and read sets relate: an import whose items are all bases of one run's overlays is
    `hidden` (it opens through that run), and run.json can set "hidden": true and "canonical": "<dataset>" itself."""
    rows = IX.query(
        """SELECT ds.d, ds.title, ds.n_items, ds.n_tok_items, ds.indexed_at,
                  (SELECT count(*) FROM reads r WHERE r.d = ds.d) n_reads
           FROM datasets ds ORDER BY ds.d"""
    )
    rel = IX.query("SELECT d, base_d, count(DISTINCT base_i) n FROM bases GROUP BY ALL")
    used_by, reads_from = {}, {}
    for r in rel:
        used_by.setdefault(r["base_d"], []).append({"d": r["d"], "n": r["n"]})
        reads_from.setdefault(r["d"], []).append({"d": r["base_d"], "n": r["n"]})
    sample = IX.query(  # one item per dataset, for a sample Docent link (its collection)
        f"SELECT it.d, {LINK_COLS} FROM items it {LINKS_JOIN} QUALIFY row_number() OVER (PARTITION BY it.d ORDER BY it.line) = 1"
    )
    first = {r["d"]: links_row(r) for r in sample}
    for r in rows:
        p = run_dir(r["d"])
        info = json.loads((p / "run.json").read_text()) if (p / "run.json").exists() else {}
        r["mtime"] = int((p / "items.jsonl").stat().st_mtime) if (p / "items.jsonl").exists() else None
        r["used_by"], r["reads_from"] = used_by.get(r["d"], []), reads_from.get(r["d"], [])
        covered = sum(x["n"] for x in r["used_by"])
        r["hidden"] = bool(info.get("hidden")) or (bool(r["used_by"]) and covered >= (r["n_items"] or 0))
        r["canonical"] = info.get("canonical") or (r["used_by"][0]["d"] if r["hidden"] and r["used_by"] else None)
        parts = r["d"].split("/")
        r["experiment"] = info.get("experiment") or (parts[1] if parts[0] == "runs" and len(parts) > 2 else None)
        r["docent"] = next((v for k, v in first.get(r["d"], {}).items() if "docent" in (k + str(v)).lower()), None)
    return rows


@app.get("/api/run")
def run(d: str) -> JSONResponse:
    p = run_dir(d)
    info = json.loads((p / "run.json").read_text()) if (p / "run.json").exists() else {"title": p.name}
    items = IX.query(f"SELECT it.i, it.grp, it.labels, it.calls, it.ref, it.n_tok, it.n_reads, {LINK_COLS} FROM items it {LINKS_JOIN} "
                     "WHERE it.d = ? ORDER BY it.line", [d])  # fmt: skip
    out = [
        {
            "id": r["i"],
            "group": r["grp"],
            "labels": json.loads(r["labels"]),
            "calls": json.loads(r["calls"]),
            "links": links_row(r),
            "tokens": {"ref": r["ref"], "n": r["n_tok"], "reads": r["n_reads"]} if r["ref"] else None,
        }
        for r in items
    ]
    return JSONResponse({"id": d, "run": info, "items": out})


# item columns + its overlay base's, so an overlay inherits the base's Docent link; use with LINKS_JOIN on alias `it`
LINK_COLS = "it.links, it.src, it.fp, bi.links blinks, bi.src bsrc, bi.fp bfp"
LINKS_JOIN = "LEFT JOIN bases b ON b.d = it.d AND b.i = it.i LEFT JOIN items bi ON bi.d = b.base_d AND bi.i = b.base_i"


def links_row(r: dict) -> dict:
    """An item's links with Docent resolved (its own, its source row's, its transcript's, or its overlay base's)."""
    out = IX.docent.links(json.loads(r.get("links") or "{}"), r.get("src"), r.get("fp"))
    if r.get("blinks") is not None:
        base = IX.docent.links(json.loads(r["blinks"]), r.get("bsrc"), r.get("bfp"))
        out = {**{k: v for k, v in base.items() if "docent" in (k + str(v)).lower()}, **out}
    return out


def raw_item(d: str, i: str) -> dict:
    rows = IX.query("SELECT off, len FROM items WHERE d = ? AND i = ?", [d, i])
    if not rows:
        raise HTTPException(404, f"no item {i!r} in {d}")
    p = run_dir(d)
    with open(p / "items.jsonl", "rb") as f:
        f.seek(rows[0]["off"])
        it = json.loads(f.read(rows[0]["len"]))
    if isinstance(it.get("env"), dict):  # inline env: resolve now, the page asks for files by id
        eid, files = resolve_env(p, it["env"])
        it["env"], it["env_files"] = eid, files
    return it


def resolved_item(d: str, i: str, depth: int = 0) -> dict:
    """An item with its overlay chain merged in (base of a base ..., at most 4 deep)."""
    it = raw_item(d, i)
    b = it.get("base")
    if isinstance(b, dict) and depth < 4:
        try:
            return merge_overlay(it, resolved_item(b["d"], b["i"], depth + 1), b["d"])
        except HTTPException:
            it["base_missing"] = b
    return it


@app.get("/api/item")
def item(d: str, i: str) -> JSONResponse:
    it = resolved_item(d, i)
    row = IX.query(f"SELECT {LINK_COLS} FROM items it {LINKS_JOIN} WHERE it.d = ? AND it.i = ?", [d, i])
    # overlays from other runs that annotate this item (a run item can itself stand on an O-lens import)
    it["annotations"] = IX.query("SELECT d, i, labels FROM bases WHERE base_d = ? AND base_i = ? ORDER BY d, i", [d, i])
    if row:
        links = {**(it.get("links") or {}), **links_row(row[0])}
        it["links"] = {k: v for k, v in links.items() if not any(points_at(v, c["d"], c["i"]) for c in it.get("chain") or [])}
    return JSONResponse(it)


@app.get("/api/env")
def env(d: str, e: str) -> JSONResponse:
    if not re.fullmatch(r"[0-9a-f]{6,64}", e):
        raise HTTPException(400, "bad env id")
    return JSONResponse(resolve_env(run_dir(d), e)[1])


@app.get("/api/tokens")
def tokens(d: str, ref: str) -> Response:
    if not re.fullmatch(r"[0-9A-Za-z_-]{1,64}", ref):
        raise HTTPException(400, "bad ref")
    return Response(json.dumps(load_tokens(run_dir(d), ref), separators=(",", ":")), media_type="application/json")


@app.post("/api/reindex")
def reindex(d: str | None = None) -> dict:
    threading.Thread(target=IX.update, kwargs={"only": d}, daemon=True).start()
    return {"started": d or "all changed runs"}


@app.get("/api/facets")
def facets(d: str = "") -> dict:
    ds = [x for x in d.split(",") if x]
    w, p = (f"d IN ({','.join('?' * len(ds))})", ds) if ds else ("TRUE", [])
    return {
        "tools": IX.query(f"SELECT tool, count(*) n FROM reads WHERE {w} GROUP BY 1 ORDER BY 2 DESC", p),
        "secs": IX.query(f"SELECT sec, count(*) n FROM positions WHERE {w} GROUP BY 1 ORDER BY 2 DESC LIMIT 100", p),
        "groups": IX.query(f"SELECT grp, count(*) n FROM items WHERE {w} AND grp IS NOT NULL GROUP BY 1 ORDER BY 1 LIMIT 500", p),
        "labels": IX.query(  # yes/no labels and labels with few values: the ones worth filtering on
            f"""WITH c AS (SELECT k, v, any_value(t) t, count(*) n FROM labels WHERE {w} GROUP BY k, v),
                kk AS (SELECT k FROM c GROUP BY k HAVING count(*) <= 30)
                SELECT c.* FROM c JOIN kk USING (k) ORDER BY k, v""",
            p,
        ),
    }


class Filters:
    """Query-string filters shared by /api/search and /api/csv."""

    def __init__(self, q: str, case: str, scope: str, novel: str, d: str, tool: str, sec: str, group: str, label: list[str]):
        if not q:
            raise HTTPException(400, "empty regex")
        try:
            re.compile(q)
        except re.error as err:
            raise HTTPException(400, f"bad regex: {err}") from err
        self.rx = q if case == "1" else f"(?i){q}"
        self.scope, self.novel = scope, novel
        self.item_conds, self.item_params = [], []  # over items it
        for col, val in (("it.d", d), ("it.grp", group)):
            vals = [v for v in val.split(",") if v]
            if vals:
                self.item_conds.append(f"{col} IN ({','.join('?' * len(vals))})")
                self.item_params += vals
        for kv in label:
            k, _, v = kv.partition("=")
            self.item_conds.append("EXISTS (SELECT 1 FROM labels l WHERE l.d = it.d AND l.i = it.i AND l.k = ? AND l.v = ?)")
            self.item_params += [k, v]
        self.read_conds, self.read_params = [], []  # over reads r / positions p
        for col, val in (("r.tool", tool), ("p.sec", sec)):
            vals = [v for v in val.split(",") if v]
            if vals:
                self.read_conds.append(f"{col} IN ({','.join('?' * len(vals))})")
                self.read_params += vals

    def from_sql(self) -> str:
        return READS_FROM + (TEXTS_JOIN if self.novel == "transcript" else "")

    def items_sql(self) -> tuple[str, list]:
        return " AND ".join(self.item_conds) or "TRUE", list(self.item_params)

    def reads_sql(self) -> tuple[str, list]:
        conds, params = ["regexp_matches(r.text, ?)"], [self.rx]
        if self.novel in ("ctx", "transcript"):  # an echo: the regex also matches what the model just read or wrote next
            conds.append("NOT regexp_matches(p.ctx || p.nxt, ?)")
            params.append(self.rx)
        if self.novel == "transcript":  # needs TEXTS_JOIN (from_sql)
            conds.append("NOT regexp_matches(coalesce(tx.text, ''), ?)")
            params.append(self.rx)
        iw, ip = self.items_sql()
        return " AND ".join(conds + self.read_conds + [iw]), params + self.read_params + ip


def filters(
    q: str = "",
    case: str = "0",
    scope: str = "reads",
    novel: str = "none",
    d: str = "",
    tool: str = "",
    sec: str = "",
    group: str = "",
    label: Annotated[list[str] | None, Query()] = None,
) -> Filters:
    return Filters(q, case, scope, novel, d, tool, sec, group, label or [])


READS_FROM = "reads r JOIN positions p ON p.d = r.d AND p.i = r.i AND p.pos = r.pos JOIN items it ON it.d = r.d AND it.i = r.i"
TEXTS_JOIN = " LEFT JOIN texts tx ON tx.d = r.d AND tx.i = r.i"


@app.get("/api/search")
def search(f: Annotated[Filters, Depends(filters)], page: int = 0, per: int = 50) -> JSONResponse:
    t0 = time.time()
    iw, ip = f.items_sql()
    cur = IX.cursor()  # temp tables below are private to this connection

    def fetch(sql: str, params: list | None = None) -> list[dict]:
        res = cur.execute(sql, params or [])
        cols = [x[0] for x in res.description]
        return [dict(zip(cols, row)) for row in res.fetchall()]

    if f.scope == "text":
        cur.execute(
            f"""CREATE OR REPLACE TEMP TABLE hit_items AS SELECT tx.d, tx.i FROM texts tx JOIN items it ON it.d = tx.d AND it.i = tx.i
                WHERE regexp_matches(tx.text, ?) AND {iw}""",
            [f.rx, *ip],
        )
        total = fetch("SELECT count(*) items FROM hit_items")[0]
        rows = fetch(
            f"""SELECT h.d, h.i, it.grp, it.labels, {LINK_COLS},
                      regexp_extract(tx.text, '(?s)(.{{0,160}}(' || ? || ').{{0,160}})', 1) snippet
               FROM (SELECT * FROM hit_items ORDER BY d, i LIMIT ? OFFSET ?) h
               JOIN items it ON it.d = h.d AND it.i = h.i {LINKS_JOIN} JOIN texts tx ON tx.d = h.d AND tx.i = h.i ORDER BY h.d, h.i""",
            [f.rx, per, page * per],
        )
    else:
        rw, rp = f.reads_sql()
        cols = "r.d, r.i, r.pos, r.tool, r.label, r.text"
        cur.execute(f"CREATE OR REPLACE TEMP TABLE hit_reads AS SELECT {cols} FROM {f.from_sql()} WHERE {rw}", rp)
        cur.execute("CREATE OR REPLACE TEMP TABLE hit_items AS SELECT DISTINCT d, i FROM hit_reads")
        total = fetch("SELECT count(*) reads, count(DISTINCT (d, i, pos)) positions, count(DISTINCT (d, i)) items FROM hit_reads")[0]
        rows = fetch(
            f"""WITH hp AS (SELECT d, i, pos, list({{'tool': tool, 'label': label, 'text': text}} ORDER BY tool, label) reads
                           FROM hit_reads GROUP BY d, i, pos ORDER BY d, i, pos LIMIT ? OFFSET ?)
               SELECT hp.d, hp.i, hp.pos, p.sec, p.token, right(p.ctx, 300) ctx, p.nxt, it.grp, it.labels, {LINK_COLS}, hp.reads
               FROM hp JOIN positions p ON p.d = hp.d AND p.i = hp.i AND p.pos = hp.pos
               JOIN items it ON it.d = hp.d AND it.i = hp.i {LINKS_JOIN} ORDER BY hp.d, hp.i, hp.pos""",
            [per, page * per],
        )
    # per (dataset, group): items in the filtered slice, how many have a hit, and each yes/no label's rate with vs overall
    slices = fetch(
        f"""WITH s AS (SELECT it.d, it.grp, it.i, (h.i IS NOT NULL) hit FROM items it LEFT JOIN hit_items h ON h.d = it.d AND h.i = it.i
                       WHERE {iw} AND it.d IN (SELECT DISTINCT d FROM hit_items))
            SELECT d, grp, count(*) items, sum(hit::int) hit_items FROM s GROUP BY ALL HAVING sum(hit::int) > 0
            ORDER BY hit_items DESC, d, grp LIMIT 300""",
        ip,
    )
    rates = fetch(
        f"""WITH s AS (SELECT it.d, it.grp, it.i, (h.i IS NOT NULL) hit FROM items it LEFT JOIN hit_items h ON h.d = it.d AND h.i = it.i
                       WHERE {iw} AND it.d IN (SELECT DISTINCT d FROM hit_items))
            SELECT s.d, s.grp, l.k, avg((l.v = 'true')::int) rate, avg(CASE WHEN s.hit THEN (l.v = 'true')::int END) rate_hit
            FROM s JOIN labels l ON l.d = s.d AND l.i = s.i WHERE l.t = 'bool' GROUP BY ALL""",
        ip,
    )
    for r in rows:
        r["labels"], r["links"] = json.loads(r["labels"]), links_row(r)
        for k in ("src", "fp", "blinks", "bsrc", "bfp"):
            r.pop(k, None)
    by = {(r["d"], r["grp"]): r for r in slices}
    for r in rates:
        if (r["d"], r["grp"]) in by:
            by[(r["d"], r["grp"])].setdefault("rates", {})[r["k"]] = [r["rate_hit"], r["rate"]]
    return JSONResponse({"total": total, "slices": slices, "rows": rows, "rx": f.rx, "secs": round(time.time() - t0, 2)})


@app.get("/api/csv")
def export(f: Annotated[Filters, Depends(filters)]) -> StreamingResponse:
    rw, rp = f.reads_sql()
    rows = IX.query(
        f"""SELECT r.d dataset, r.i item, it.grp "group", r.pos, p.sec, p.token, r.tool, r.label, r.text, p.ctx, p.nxt, it.labels
            FROM {f.from_sql()} WHERE {rw} ORDER BY r.d, r.i, r.pos LIMIT 500000""",
        rp,
    )
    buf = io.StringIO()
    if rows:
        wr = csv.DictWriter(buf, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    headers = {"Content-Disposition": "attachment; filename=expview_search.csv"}
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv", headers=headers)


def rescan(every: int) -> None:
    while True:
        try:
            IX.update()
        except Exception as err:  # noqa: BLE001 - keep serving; the next pass retries
            print(f"rescan failed: {err!r}", flush=True)
        time.sleep(every)


def main() -> None:
    global IX
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=REPO / "data/expview/index.duckdb")
    ap.add_argument("--host", action="append", default=[])
    ap.add_argument("--port", type=int, default=8830)
    ap.add_argument("--rescan", type=int, default=120, help="seconds between checks for new or changed runs")
    a = ap.parse_args()
    IX = Index(a.db)
    threading.Thread(target=rescan, args=(a.rescan,), daemon=True).start()
    socks = []
    for h in a.host or ["127.0.0.1"]:
        fam = socket.AF_INET6 if ":" in h else socket.AF_INET
        s = socket.socket(fam, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if fam == socket.AF_INET6:
            s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        s.bind((h, a.port))
        socks.append(s)
    print(f"expview on {', '.join(a.host or ['127.0.0.1'])} port {a.port}, index {a.db}", flush=True)
    uvicorn.Server(uvicorn.Config(app, log_level="warning")).run(sockets=socks)


if __name__ == "__main__":
    main()
