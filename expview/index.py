# /// script
# requires-python = ">=3.12"
# dependencies = ["duckdb>=1.1", "pyarrow>=17"]
# ///
"""The search index behind expview/server.py: one DuckDB file over every run under the roots, updated per run when its
files change. The run dirs stay the source of truth; the index only holds what search and the item lists need.

Tables (d = dataset id, i = item id):
  datasets   d, path, title, n_items, n_tok_items, sig (files' size+mtime, to spot changes), indexed_at
  items      d, i, line, off, len (byte span in items.jsonl), grp, labels / calls / links (JSON text), ref, n_tok, n_reads,
             src (ENV/RUN:LINE of the source row), fp (fingerprint of the subject's replies): Docent links resolve from these
  labels     d, i, k, v, t ('bool' | 'num' | 'str'): one row per item label, for filters and per-label rates
  positions  d, i, pos, sec, token, ctx (40 tokens before + the token), nxt (next 16 tokens): one row per read token
  reads      d, i, pos, tool, label, text (the read's lines joined by newlines)
  bases      d, i, base_d, base_i, labels: overlay items (README "Overlays") -> the item they annotate
  texts      d, i, text: everything the item shows (call messages, then the token stream), for transcript search and
             "new vs the whole transcript"

Rebuild (e.g. to compact: DuckDB keeps the space of re-indexed runs) into a NEW file while the server keeps serving the
old one, then restart the server on it:

    uv run --script expview/index.py --db data/expview/index.new.duckdb
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).parent))
import expview
from expview import discover, item_tokens, load_tokens

CTX, NXT = 40, 16
SCHEMA = {
    "datasets": "d VARCHAR, path VARCHAR, title VARCHAR, n_items BIGINT, n_tok_items BIGINT, sig VARCHAR, indexed_at VARCHAR",
    "items": "d VARCHAR, i VARCHAR, line BIGINT, off BIGINT, len BIGINT, grp VARCHAR, labels VARCHAR, calls VARCHAR, links VARCHAR, "
    "ref VARCHAR, n_tok BIGINT, n_reads BIGINT, src VARCHAR, fp VARCHAR",
    "labels": "d VARCHAR, i VARCHAR, k VARCHAR, v VARCHAR, t VARCHAR",
    "positions": "d VARCHAR, i VARCHAR, pos BIGINT, sec VARCHAR, token VARCHAR, ctx VARCHAR, nxt VARCHAR",
    "reads": "d VARCHAR, i VARCHAR, pos BIGINT, tool VARCHAR, label VARCHAR, text VARCHAR",
    "texts": "d VARCHAR, i VARCHAR, text VARCHAR",
    "bases": "d VARCHAR, i VARCHAR, base_d VARCHAR, base_i VARCHAR, labels VARCHAR",
}
FLUSH = 200_000  # rows buffered per table before an insert


def signature(run: Path) -> str:
    parts = []
    for f in (run / "items.jsonl", run / "run.json"):
        if f.exists():
            s = f.stat()
            parts.append(f"{s.st_size}:{s.st_mtime_ns}")
    tk = run / "tokens"
    if tk.is_dir():
        parts.append(str(tk.stat().st_mtime_ns))
    return "|".join(parts)


JUDGE_RX = re.compile(r"judge|grader|grading|classif|monitor|scor|rater|critic|verdict", re.IGNORECASE)  # as in static/index.html
DOCENT = "https://docent.transluce.org"


def fingerprint(messages) -> str | None:
    """sha1 of the assistant turns' text: the same transcript in a rows.jsonl row and an expview call gets the same value."""
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except ValueError:
            return None
    parts = [content_text(m.get("content")) for m in messages or [] if isinstance(m, dict) and m.get("role") == "assistant"]
    return hashlib.sha1("\x00".join(parts).encode()).hexdigest()[:20] if any(x.strip() for x in parts) else None


def source_of(it: dict) -> str | None:
    """ENV/RUN:LINE of the rows.jsonl row an item shows, when the item says so (extra.source, or run_dir + row_index)."""
    ex = it.get("extra") if isinstance(it.get("extra"), dict) else {}
    m = re.search(r"(?:^|/)([^/:]+/[^/:]+):(\d+)$", str(ex.get("source") or ""))
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    rd, row, sf = ex.get("run_dir"), ex.get("row_index"), ex.get("source_file")
    if isinstance(rd, str) and isinstance(row, int) and sf in (None, "rows.jsonl"):
        return f"{rd.removeprefix('runs/').strip('/')}:{row}"
    return None


def subject_fp(it: dict) -> str | None:
    calls = it.get("calls") or []
    c = next((c for c in calls if not JUDGE_RX.search(str(c.get("name") or ""))), None)
    return fingerprint(c.get("messages")) if c else None


class Docent:
    """Docent run URLs for ENV/RUN:LINE and for transcript fingerprints, from every docent.json (scripts/docent_upload.py) under
    the data dirs: <data>/runs, <data>/hf_stage*/runs, <data>/olens_search/docent/runs. Reloaded when any docent.json changes."""

    def __init__(self, data: Path):
        self.data, self.sig, self.runs, self.fps = data, None, {}, {}

    def files(self) -> list[Path]:
        roots = [self.data, *sorted(self.data.glob("hf_stage*")), self.data / "olens_search/docent"]
        return [f for r in roots for f in sorted(r.glob("runs/*/*/docent.json"))]

    def refresh(self) -> None:
        fs = self.files()
        sig = hashlib.sha1("".join(f"{f}{f.stat().st_mtime_ns}" for f in fs).encode()).hexdigest()
        if sig == self.sig:
            return
        runs, fps = {}, {}
        for f in fs:
            key = f"{f.parent.parent.name}/{f.parent.name}"
            if key in runs:
                continue
            try:
                doc = json.loads(f.read_text())
            except ValueError:
                continue
            runs[key] = (doc["collection_id"], doc.get("runs") or [])
            rows = f.parent / "rows.jsonl"
            if rows.exists():
                for n, line in enumerate(rows.open()):
                    if n < len(runs[key][1]) and runs[key][1][n] and line.strip():
                        fp = fingerprint(json.loads(line).get("messages"))
                        if fp:
                            fps.setdefault(fp, f"{key}:{n}")
        self.runs, self.fps, self.sig = runs, fps, sig
        print(f"docent: {len(runs)} uploaded runs, {len(fps)} transcript fingerprints", flush=True)

    def url(self, src: str | None) -> str | None:
        key, _, n = (src or "").rpartition(":")
        cid, ids = self.runs.get(key, (None, []))
        return f"{DOCENT}/dashboard/{cid}/agent_run/{ids[int(n)]}" if cid and n.isdigit() and int(n) < len(ids) and ids[int(n)] else None

    def links(self, links: dict, src: str | None, fp: str | None) -> dict:
        """An item's links with "Docent" filled in when it has none and its source row or transcript was uploaded."""
        if any("docent" in (k + str(v)).lower() for k, v in links.items()):
            return links
        u = self.url(src) or self.url(self.fps.get(fp or ""))
        return {**links, "Docent": u} if u else links


def links_of(it: dict) -> dict:
    out = dict(it.get("links") or {})
    ex = it.get("extra") or {}
    for k in ("docent", "docent_url"):
        if isinstance(ex.get(k), str) and ex[k].startswith("http"):
            out.setdefault("Docent", ex[k])
    return out


def content_text(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return "" if c is None else json.dumps(c)


def call_text(c: dict) -> str:
    out = []
    for m in c.get("messages") or []:
        out.append(m.get("reasoning") or "")
        out.append(content_text(m.get("content")))
        out += [f"{t.get('name')} {t.get('arguments')}" for t in m.get("tool_calls") or []]
    return "\n".join(x for x in out if x)


def label_type(v) -> str:
    return "bool" if isinstance(v, bool) else "num" if isinstance(v, (int, float)) else "str"


def label_str(v) -> str:
    return ("true" if v else "false") if isinstance(v, bool) else str(v)


def section_of(sections: list, n: int) -> list[str]:
    out = [""] * n
    for name, a, b in sections or []:
        for p in range(max(0, a), min(n, b)):
            out[p] = name
    return out


class Index:
    def __init__(self, db: Path):
        db.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(db))
        self.lock = threading.Lock()  # one writer at a time
        self.errors: dict[str, str] = {}  # dataset -> why its last indexing failed
        self.progress = {"state": "idle", "current": None, "done": 0, "todo": 0}
        for name, cols in SCHEMA.items():
            self.con.execute(f"CREATE TABLE IF NOT EXISTS {name} ({cols})")
        for col in ("src", "fp"):  # added 2026-10-25; older index files lack them until rebuilt
            self.con.execute(f"ALTER TABLE items ADD COLUMN IF NOT EXISTS {col} VARCHAR")
        self.docent = Docent(Path(os.environ.get("EXPVIEW_DATA") or expview.roots().get("runs", expview.REPO / "data/runs")).parent)
        self.paths: dict[str, Path] = {d: Path(p) for d, p in self.con.execute("SELECT d, path FROM datasets").fetchall()}

    @property
    def status(self) -> dict:
        return {**self.progress, "errors": self.errors}

    def cursor(self) -> duckdb.DuckDBPyConnection:
        """A connection of its own for one request: DuckDB readers run concurrently with each other and with indexing."""
        return self.con.cursor()

    def query(self, sql: str, params: list | None = None) -> list[dict]:
        cur = self.cursor()
        res = cur.execute(sql, params or [])
        cols = [x[0] for x in res.description]
        return [dict(zip(cols, row)) for row in res.fetchall()]

    def update(self, only: str | None = None) -> None:
        """Re-index every run whose files changed (or just `only`), drop runs that disappeared."""
        found = discover()
        try:
            self.docent.refresh()
        except Exception as err:  # noqa: BLE001 - links are a convenience; indexing must go on
            print(f"docent refresh failed: {err!r}", flush=True)
        expview._REFS.clear()  # overlays look up their base items' tokens afresh
        known = {r["d"]: r["sig"] for r in self.query("SELECT d, sig FROM datasets")}
        todo = [d for d, p in found.items() if (only is None or d == only) and (known.get(d) != signature(p) or only == d)]
        gone = [] if only else [d for d in known if d not in found]
        self.progress.update(state="indexing", done=0, todo=len(todo))
        for d in gone:
            self.drop(d)
        for d in sorted(todo):
            self.progress["current"] = d
            t0 = time.time()
            try:
                self.index_run(d, found[d])
                self.errors.pop(d, None)
                print(f"indexed {d} in {time.time() - t0:.1f}s", flush=True)
            except Exception as err:  # noqa: BLE001 - one broken run must not stop the others
                self.errors[d] = repr(err)
                print(f"failed {d}: {err!r}", flush=True)
            self.progress["done"] = int(self.progress["done"] or 0) + 1
        self.paths = {r["d"]: Path(r["path"]) for r in self.query("SELECT d, path FROM datasets")}
        self.progress.update(state="idle", current=None)

    def drop(self, d: str) -> None:
        with self.lock:
            cur = self.cursor()
            for t in SCHEMA:
                cur.execute(f"DELETE FROM {t} WHERE d = ?", [d])

    def index_run(self, d: str, run: Path) -> None:
        sig = signature(run)
        info = json.loads((run / "run.json").read_text()) if (run / "run.json").exists() else {}
        bufs: dict[str, dict[str, list]] = {t: {c.split()[0]: [] for c in SCHEMA[t].split(", ")} for t in SCHEMA}
        staged: list[tuple[str, pa.Table]] = []

        def flush(table: str) -> None:
            if bufs[table]["d"]:
                staged.append((table, pa.table(bufs[table])))
                bufs[table] = {k: [] for k in bufs[table]}

        def row(table: str, **kw) -> None:
            for k, v in kw.items():
                bufs[table][k].append(v)
            if len(bufs[table]["d"]) >= FLUSH:
                flush(table)

        n_items = n_tok = 0
        off = 0
        with open(run / "items.jsonl", "rb") as f:
            for line_no, raw in enumerate(f):
                start, off = off, off + len(raw)
                if not raw.strip():
                    continue
                it = json.loads(raw)
                i, n_items = str(it["id"]), n_items + 1
                calls = [{"name": c.get("name"), "model": c.get("model"), "n": len(c.get("messages") or [])} for c in it.get("calls") or []]
                labels = it.get("labels") or {}
                tk = it.get("tokens") or None
                text = "\n\n".join(call_text(c) for c in it.get("calls") or [])
                n_reads = 0
                b, T = it.get("base"), None
                if isinstance(b, dict):  # overlay: index its own reads / notes against the base transcript
                    row("bases", d=d, i=i, base_d=str(b.get("d")), base_i=str(b.get("i")), labels=json.dumps(labels))
                    extra = {str(p): list(rs) for p, rs in (it.get("reads") or {}).items()}
                    for nt in it.get("notes") or []:
                        if nt.get("pos") is not None:
                            extra.setdefault(str(nt["pos"]), []).append({"tool": "note", "lines": [[nt.get("text", "")]]})
                    if extra:
                        base_T = item_tokens(str(b.get("d")), str(b.get("i")))
                        if base_T:
                            T = {**base_T, "reads": extra}
                elif tk and tk.get("ref"):
                    n_tok += 1
                    try:
                        T = load_tokens(run, tk["ref"])
                    except (OSError, ValueError):
                        T = None
                if T:
                    toks = [str(x) for x in T.get("tokens") or []]
                    secs = section_of(T.get("sections"), len(toks))
                    if not isinstance(b, dict):  # an overlay's transcript is its base's, already searchable there
                        text += "\n\n" + "".join(toks)
                    for p_str, rs in (T.get("reads") or {}).items():
                        p = int(p_str)
                        if p >= len(toks):
                            continue
                        row(
                            "positions",
                            d=d,
                            i=i,
                            pos=p,
                            sec=secs[p],
                            token=toks[p],
                            ctx="".join(toks[max(0, p - CTX) : p + 1]),
                            nxt="".join(toks[p + 1 : p + 1 + NXT]),
                        )
                        for r in rs:
                            lines = "\n".join(str(x[0]) if isinstance(x, list) and x else str(x) for x in r.get("lines") or [])
                            row("reads", d=d, i=i, pos=p, tool=str(r.get("tool")), label=r.get("label"), text=lines)
                            n_reads += 1
                row(
                    "items",
                    d=d,
                    i=i,
                    line=line_no,
                    off=start,
                    len=len(raw),
                    grp=None if it.get("group") is None else str(it.get("group")),
                    labels=json.dumps(labels),
                    calls=json.dumps(calls),
                    links=json.dumps(links_of(it)),
                    src=source_of(it),
                    fp=subject_fp(it),
                    ref=tk.get("ref") if tk else None,
                    n_tok=tk.get("n") if tk else None,
                    n_reads=n_reads,
                )
                for k, v in labels.items():
                    if v is not None and not isinstance(v, (dict, list)):
                        row("labels", d=d, i=i, k=str(k), v=label_str(v), t=label_type(v))
                row("texts", d=d, i=i, text=text)
        for t in SCHEMA:
            if t != "datasets":
                flush(t)
        meta = pa.table(
            {
                "d": [d],
                "path": [str(run)],
                "title": [info.get("title") or run.name],
                "n_items": [n_items],
                "n_tok_items": [n_tok],
                "sig": [sig],
                "indexed_at": [datetime.now(UTC).isoformat(timespec="seconds")],
            }
        )
        staged.append(("datasets", meta))
        with self.lock:  # swap the run's rows in one transaction so searches never see half a run
            cur = self.cursor()
            cur.execute("BEGIN")
            try:
                for t in SCHEMA:
                    cur.execute(f"DELETE FROM {t} WHERE d = ?", [d])
                for t, tab in staged:
                    cur.register("staged", tab)
                    cur.execute(f"INSERT INTO {t} BY NAME SELECT * FROM staged")
                    cur.unregister("staged")
                cur.execute("COMMIT")
            except Exception:
                cur.execute("ROLLBACK")
                raise


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a fresh index into a new file (the live one keeps serving).")
    ap.add_argument("--db", type=Path, required=True, help="the new index file; must not exist")
    a = ap.parse_args()
    if a.db.exists():
        sys.exit(f"{a.db} exists: build into a new path, never over the live index")
    tmp = a.db.with_name(a.db.name + ".building")
    for f in (tmp, tmp.with_name(tmp.name + ".wal")):
        f.unlink(missing_ok=True)
    t0 = time.time()
    ix = Index(tmp)
    ix.update()
    ix.con.execute("CHECKPOINT")
    ix.con.close()
    tmp.rename(a.db)
    print(f"{a.db}: {len(ix.paths)} datasets in {time.time() - t0:.0f}s, {a.db.stat().st_size / 1e9:.1f} GB; errors: {ix.errors or 'none'}")


if __name__ == "__main__":
    main()
