"""Search everything: one regex over every metamodel read, env episode, transcript and judge call in a local data mirror.

Kylie's exception (2026-10-27) to "viewers are static, no backend": the page's "Search everything" calls this server, which runs on
bramble over the local mirror of the HF dataset (data/). It indexes every run (expview.write_search: env runs under
data/envs/<env>/<run>, expview runs under data/runs/**/items.jsonl, batch reads through their expview pointers) into
CACHE/<dataset id>.jsonl.gz, rebuilding a run when any of its files is newer than its index, holds them in memory, and answers
/api/search with the hits (deep links are built by the page). It also serves SITE (a folder holding only the two pages:
SITE/index.html = environments/_lib/page.html, SITE/expview/index.html = expview/static/index.html), so the page and the API
share an origin: an https page (GitHub Pages) can't call an http server, so the button shows only on the served copy, or
where localStorage "expview:search_server" names a reachable server.

    uv run python expview/search_server.py --data data --site SITE --port 8840 [--cache DIR] [--host ::]
    curl 'http://localhost:8840/api/search?query=%5Cbgrader'   # what the page fetches
    curl -X POST http://localhost:8840/api/reload              # re-scan the mirror now (it also does so every 10 min)
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import gzip
import io
import json
import mmap
import os
import re
import subprocess
import sys
import threading
import time
from array import array
from concurrent.futures import ProcessPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).parent))
import expview

PER_PAGE = 50
SEP = "\x00"


IMPORTS = "imports/celeste"  # Celeste's GLM-5.3 O-lens readouts (importers/celeste.py): local only, so the server serves their files


def exp_of(d: str) -> str:
    """The experiment a dataset id belongs to: runs/<exp>/…, envs/<exp>/…; Celeste's imports are celeste/<config>."""
    ps = d.split("/")
    return f"celeste/{ps[-1]}" if d.startswith(IMPORTS + "/") else ps[1]


def discover(data: Path, imports: Path | None) -> dict[str, Path]:
    """dataset id -> run dir: env runs as envs/<env>/<run>, expview runs as runs/<experiment>/<run>[/<name>] (the page's ids),
    and Celeste's imports as imports/celeste/<source>/<config>."""
    out = {}
    for items in sorted(imports.rglob("items.jsonl")) if imports and imports.is_dir() else []:
        out[f"{IMPORTS}/{items.parent.relative_to(imports)}"] = items.parent
    for ix in sorted((data / "envs").glob("*/*/episodes/index.json")):
        run = ix.parent.parent
        out[f"envs/{run.parent.name}/{run.name}"] = run
    for items in sorted((data / "runs").rglob("items.jsonl")):
        rel = items.parent.relative_to(data)
        if "expview" in rel.parts:
            out["/".join(p for p in rel.parts if p != "expview")] = items.parent
    return out


def sources(run: Path) -> list[Path]:
    """The files a run's index depends on (batch reads live beside the expview dir, so the run dir's parent counts)."""
    if (run / "episodes").is_dir():
        return [*run.glob("episodes/*"), *run.glob("labels/*"), run / "meta.json"]
    fs = [run / "items.jsonl", *run.glob("*.jsonl")]
    return fs + [*run.parent.glob("*.jsonl")] if run.name == "expview" else fs


def build_one(args):
    d, run, out = args
    try:
        return d, expview.write_search(run, out), None
    except Exception as e:  # noqa: BLE001
        return d, 0, repr(e)[:300]


class Corpus:
    def __init__(self, data: Path, imports: Path | None, cache: Path, workers: int):
        self.data, self.imports, self.cache, self.workers = data, imports, cache, workers
        self.files: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.errors: dict[str, str] = {}

    def refresh(self):
        runs = discover(self.data, self.imports)
        stale = []
        for d, run in runs.items():
            out = self.cache / (d + ".jsonl.gz")
            newest = max((f.stat().st_mtime for f in sources(run) if f.exists()), default=0)
            if not out.exists() or out.stat().st_mtime < newest:
                out.parent.mkdir(parents=True, exist_ok=True)
                stale.append((d, run, out))
        if stale:
            print(f"indexing {len(stale)} runs", flush=True)
            with ProcessPoolExecutor(self.workers) as ex:
                for k, (d, n, err) in enumerate(ex.map(build_one, stale)):
                    if err:
                        self.errors[d] = err
                        print(f"  {d}: {err}", flush=True)
                    if k % 50 == 0:
                        print(f"  {k}/{len(stale)}", flush=True)
        changed = {d for d, *_ in stale}
        files = {}
        for d in runs:
            out = self.cache / (d + ".jsonl.gz")
            if not out.exists():
                continue
            files[d] = self.files[d] if d in self.files and d not in changed else load(d, out)
            files[d]["model"] = model_label(runs[d])
        with self.lock:
            self.files = files
        ctypes.CDLL("libc.so.6").malloc_trim(0)  # loading churns through every string once; hand the freed heap back
        print(f"{len(files)} runs, {sum(len(f['items']) for f in files.values())} items loaded", flush=True)


def model_label(run: Path) -> str | None:
    """An env run's target as the env viewer names it: glm-5.3 · thinking off (expview runs: None, their groups say it)."""
    if not (run / "meta.json").exists() or not (run / "episodes").is_dir():
        return None
    t = json.loads((run / "meta.json").read_text()).get("target") or {}
    t = t if isinstance(t, dict) else {"id": t}
    name = str(t.get("model") or t.get("id") or "?").split("@")[0].split("/")[-1].lower()
    return name + {False: " · thinking off", True: " · thinking on"}.get(t.get("thinking"), "")


def load(d: str, path: Path) -> dict:
    """A search file -> its distinct strings in a NUL-separated file (read back through mmap by offset), and items whose token
    stream (t) and reads (rp / rt / rl / rx: position, tool, label, text) are arrays of string indices; facets counted once here."""
    S: list[str] = []

    def s(x):
        if isinstance(x, int):
            return x
        S.append(x)
        return len(S) - 1

    items, tools, secs, groups, labs = [], {}, set(), {}, {}
    with gzip.open(path, "rt") as f:
        hdr = json.loads(f.readline())
        for line in f:
            x = json.loads(line)
            if "t" in x:
                x["t"] = array("i", (s(t) for t in x["t"]))
            if r := x.pop("r", None):
                x["rp"], x["rt"], x["rl"], x["rx"] = (
                    array("i", (rd[0] for rd in r)),
                    array("i"),
                    [sys.intern(str(rd[2] or "")) for rd in r],
                    array("i"),
                )
                for rd in r:
                    x["rt"].append(s(rd[1]))
                    x["rx"].append(s(rd[3]))
            for c in x.get("c") or []:
                c[1] = array("i", (s(t) for t in c[1]))
            for j in x.get("rt") or []:
                tools[S[j]] = tools.get(S[j], 0) + 1
            for sec in x.get("s") or []:
                secs.add(sec[0])
            groups[x.get("g")] = groups.get(x.get("g"), 0) + 1
            for k, v in (x.get("l") or {}).items():
                if v is not None and not isinstance(v, (dict, list)):
                    m = labs.setdefault(k, {})
                    m[label_str(v)] = m.get(label_str(v), 0) + 1
            items.append(x)
    for x in items:  # each token stream joined is a string too, so a match across tokens is found like any other
        if "t" in x:
            S.append("".join(S[j] for j in x["t"]))
            x["tt"] = len(S) - 1
    # the strings go to disk, one per NUL-terminated record (ripgrep --null-data: record n = string n - 1), read back by offset
    spath, starts, n = path.with_name(path.name.removesuffix(".jsonl.gz") + ".strings"), array("q"), 0
    with open(spath, "wb") as f:
        for t in S:
            b = t.replace(SEP, " ").encode() + b"\0"
            starts.append(n)
            n += len(b)
            f.write(b)
    starts.append(n)
    hdr["kind"] = hdr.get("kind") or "expview"
    fh = open(spath, "rb")  # noqa: SIM115  (held open for the mmap)
    return {
        "d": d,
        "exp": exp_of(d),
        "hdr": hdr,
        "strings": str(spath),
        "mm": mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) if n else b"",
        "starts": starts,
        "items": items,
        "facets": {"reader": tools, "section": secs, "group": groups, "label": labs},
    }


def at(F, j: int) -> str:
    return F["mm"][F["starts"][j] : F["starts"][j + 1] - 1].decode()


def string_hits(files: list[dict], query: str, case: bool) -> dict[str, set[int]]:
    """Per run, the indices of the strings the regex matches: one ripgrep over the runs' string files (Rust regex syntax, or
    PCRE2 when it needs lookaround / backreferences)."""
    by = {F["strings"]: F["d"] for F in files}
    out: dict[str, set[int]] = {F["d"]: set() for F in files}
    for engine in ([], ["-P"]):
        cmd = [
            "rg",
            "--null-data",
            "--line-number",
            "--with-filename",
            "--no-heading",
            "-o",
            "-r",
            "",
            "-a",
            *engine,
            *([] if case else ["-i"]),
        ]
        p = subprocess.run([*cmd, "-e", query, "--", *by], capture_output=True, check=False)
        if p.returncode in (0, 1):
            for rec in p.stdout.split(b"\0"):
                if rec:
                    path, ln, _ = rec.decode().rsplit(":", 2)
                    out[by[path]].add(int(ln) - 1)
            return out
    raise re.error(p.stderr.decode().strip()[-300:])


def label_str(v):
    return ("true" if v else "false") if isinstance(v, bool) else str(v)


def search(corpus: Corpus, q: dict) -> dict:
    """The Search page's query (same keys as its URL) -> totals, the per-(run, group) table, facets and every hit, in corpus
    order: text hits {kind: "text", where, snippet} and read positions {kind: "pos", pos, ctx, token, nxt, reads}."""
    g = lambda k: (q.get(k) or [""])[0]
    query = g("query")
    query = re.sub(r"^\(\?i\)", "", query)
    rx = re.compile(query, 0 if g("case") == "1" else re.IGNORECASE)  # for snippets and the echo filters
    scope, novel, tool, sec, Fg, Fl = g("scope") or "all", g("novel"), g("reader"), g("section"), g("group"), g("label")
    lk, lv = Fl.split("=", 1) if "=" in Fl else ("", "")
    exp, ds, model = g("experiment"), {x for x in g("dataset").split(",") if x}, g("model")
    with corpus.lock:
        files = [F for F in corpus.files.values() if (not exp or F["exp"] == exp) and (not ds or F["d"] in ds)]
    files = [F for F in files if not model or F["model"] == model]
    t0 = time.time()
    HS = string_hits(files, query, g("case") == "1")
    ok = lambda x: (not Fg or x.get("g") == Fg) and (not lk or (x.get("l") or {}).get(lk) is not None and label_str(x["l"][lk]) == lv)
    hits, n_reads, n_text, n_pos, hit_items, read_items, text_items = [], 0, 0, 0, set(), set(), set()
    tools, secs, groups, labs = {}, set(), {}, {}
    slices, subs = {}, {}
    for F in files:
        fc = F["facets"]
        for k, n in fc["reader"].items():
            tools[k] = tools.get(k, 0) + n
        secs |= fc["section"]
        for k, n in fc["group"].items():
            groups[k] = groups.get(k, 0) + n
        for k, m in fc["label"].items():
            for v, n in m.items():
                labs.setdefault(k, {})[v] = labs.get(k, {}).get(v, 0) + n
        H = HS[F["d"]]
        for x in F["items"]:
            if not ok(x):
                continue
            key = (F["d"], x["i"])
            base = {"d": F["d"], "exp": F["exp"], "i": x["i"], "g": x.get("g"), "l": x.get("l") or {}}
            mine = []
            if scope != "text" and "rx" in x and H and not H.isdisjoint(x["rx"]):
                t, secs_ = x.get("t") or array("i"), x.get("s") or []
                span = lambda a, b: "".join(at(F, j) for j in t[max(0, a) : b])  # noqa: B023
                sec_at = lambda p: next((name for name, a, b in secs_ if a <= p < b), "")  # noqa: B023
                by: dict[int, dict] = {}
                echo: set[int] = set()
                for pos, ti, label, tj in zip(x["rp"], x["rt"], x["rl"], x["rx"]):
                    if tj not in H or pos in echo or (tool and at(F, ti) != tool) or (sec and sec_at(pos) != sec):
                        continue
                    if pos not in by:
                        ctx, nxt = span(pos - 40, pos + 1), span(pos + 1, pos + 17)
                        if novel and rx.search(ctx + nxt):
                            echo.add(pos)
                            continue
                        by[pos] = {
                            **base,
                            "kind": "pos",
                            "pos": pos,
                            "sec": sec_at(pos),
                            "ctx": ctx[-300:],
                            "nxt": nxt,
                            "reads": [],
                        }
                        by[pos]["token"] = at(F, t[pos]) if pos < len(t) else ""
                    by[pos]["reads"].append([at(F, ti), label, at(F, tj)])
                mine = list(by.values())
                if novel == "transcript" and mine and rx.search(item_text(F, x)):
                    mine = []
            if mine:
                n_pos += len(mine)
                n_reads += sum(len(P["reads"]) for P in mine)
                read_items.add(key)
            text = None
            if scope != "reads":
                for name, parts in x.get("c") or []:
                    j = next((j for j in parts if j in H), None)
                    if j is not None:
                        text = (name, at(F, j))
                        break
                if text is None and x.get("tt") in H:
                    text = ("transcript", at(F, x["tt"]))
                if text:
                    m = rx.search(text[1])  # None when Python's re reads the pattern differently from ripgrep
                    hits.append(
                        {
                            **base,
                            "kind": "text",
                            "where": text[0],
                            "snippet": text[1][max(0, m.start() - 160) : m.end() + 160] if m else text[1][:320],
                        }
                    )
                    n_text += 1
                    text_items.add(key)
            hits += sorted(mine, key=lambda P: P["pos"])
            hit = bool(mine or text)
            if hit:
                hit_items.add(key)
            # subgroups of an experiment: the item's group (an env episode's arm) x the env run's model; no group -> the dataset
            gk = (x.get("g") if x.get("g") is not None else None, F["model"], F["d"] if x.get("g") is None else None)
            sg = subs.setdefault(gk, {"group": gk[0], "model": gk[1], "d": gk[2], "items": 0, "hit_items": 0, "text": 0, "reads": 0})
            sg.setdefault("rates", {})
            sg["items"] += 1
            sg["hit_items"] += hit
            sg["text"] += bool(text)
            sg["reads"] += bool(mine)
            for k, v in (x.get("l") or {}).items():
                if isinstance(v, bool):
                    r = sg["rates"].setdefault(k, [0, 0, 0, 0])
                    r[0] += 1
                    r[1] += v
                    r[2] += hit
                    r[3] += hit and v
            sk = (F["d"], x.get("g"))
            s_ = slices.setdefault(sk, {"d": F["d"], "grp": x.get("g"), "items": 0, "hit_items": 0, "rates": {}})
            s_["items"] += 1
            s_["hit_items"] += hit
            for k, v in (x.get("l") or {}).items():
                if isinstance(v, bool):
                    r = s_["rates"].setdefault(k, [0, 0, 0, 0])  # n, true, n with hit, true with hit
                    r[0] += 1
                    r[1] += v
                    r[2] += hit
                    r[3] += hit and v
    table = sorted((x for x in slices.values() if x["hit_items"]), key=lambda x: (-x["hit_items"], x["d"]))[:300]
    exps: dict[str, dict] = {}
    for d, i in hit_items:
        e = exps.setdefault(exp_of(d), {"exp": exp_of(d), "items": 0, "text": 0, "reads": 0, "runs": set()})
        e["items"] += 1
        e["text"] += (d, i) in text_items
        e["reads"] += (d, i) in read_items
        e["runs"].add(d)
    by_exp = sorted(({**e, "runs": len(e["runs"])} for e in exps.values()), key=lambda e: -e["items"])
    hits.sort(key=lambda h: -exps[exp_of(h["d"])]["items"])  # stable: grouped by experiment, most hits first
    meta = {F["d"]: {"kind": F["hdr"]["kind"], "target": F["hdr"].get("target")} for F in files if F["d"] in {d for d, _ in hit_items}}
    return {
        "took": round(time.time() - t0, 2),
        "runs": len(files),
        "items": sum(len(F["items"]) for F in files),
        "reads": sum(tools.values()),
        "hit_items": len(hit_items),
        "hit_runs": len({d for d, _ in hit_items}),
        "hit_experiments": len({exp_of(d) for d, _ in hit_items}),
        "text_items": n_text,
        "read_hits": n_reads,
        "read_positions": n_pos,
        "read_items": len(read_items),
        "table": table,
        "by_experiment": by_exp,
        "by_group": sorted((x for x in subs.values() if x["hit_items"]), key=lambda x: (-x["hit_items"], str(x["group"]), str(x["model"])))[
            :500
        ]
        if exp
        else [],
        "facets": {
            "reader": sorted(tools.items(), key=lambda x: -x[1]),
            "section": sorted(secs),
            "group": sorted(((k, n) for k, n in groups.items() if k is not None), key=lambda x: str(x[0]))[:2000],
            "label": [[k, sorted(m.items())] for k, m in labs.items() if len(m) <= 30],
        },
        "experiments": sorted({F["exp"] for F in corpus.files.values()}),
        "meta": meta,
        "hits": hits,
    }


def item_text(F, x):
    return "\n\n".join(at(F, j) for j in [*(j for _, parts in x.get("c") or [] for j in parts), *([x["tt"]] if "tt" in x else [])])


CACHE: dict[tuple, dict] = {}


def cached_search(corpus: Corpus, q: dict) -> dict:
    """The last few searches, so paging and the CSV don't rerun one (an all-experiment query over dense reads takes ~10 s)."""
    key = (id(corpus.files), *sorted((k, tuple(v)) for k, v in q.items() if k != "page"))
    if key not in CACHE:
        if len(CACHE) >= 8:
            CACHE.pop(next(iter(CACHE)))
        CACHE[key] = search(corpus, q)
    return CACHE[key]


def runs_list(corpus: Corpus):
    with corpus.lock:
        return [
            {"d": F["d"], "exp": F["exp"], "kind": F["hdr"]["kind"], "target": F["hdr"].get("target"), "items": len(F["items"])}
            for F in corpus.files.values()
        ]


def serve(corpus: Corpus, site: Path, host: str, port: int):
    class H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(site), **k)

        def send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path == "/api/ping":
                    with corpus.lock:
                        n = len(corpus.files)
                    return self.send(200, json.dumps({"ok": True, "runs": n}).encode())
                if u.path.startswith(f"/api/tree/{IMPORTS}/") or u.path.startswith(f"/files/{IMPORTS}/"):
                    return self.imports_file(u.path)
                if u.path == "/api/runs":
                    return self.send(200, json.dumps(runs_list(corpus)).encode())
                if u.path in ("/api/search", "/api/csv"):
                    r = cached_search(corpus, q)
                    if u.path == "/api/csv":
                        buf = io.StringIO()
                        w = csv.writer(buf)
                        w.writerow(["dataset", "item", "group", "pos", "sec", "token", "tool", "label", "text", "ctx", "nxt", "labels"])
                        for h in r["hits"]:
                            for tool, label, text in h.get("reads") or []:
                                w.writerow(
                                    [
                                        h["d"],
                                        h["i"],
                                        h["g"],
                                        h["pos"],
                                        h["sec"],
                                        h["token"],
                                        tool,
                                        label,
                                        text,
                                        h["ctx"],
                                        h["nxt"],
                                        json.dumps(h["l"]),
                                    ]
                                )
                        return self.send(200, buf.getvalue().encode(), "text/csv")
                    page = int((q.get("page") or ["0"])[0] or 0)
                    r = {**r, "total_hits": len(r["hits"]), "hits": r["hits"][page * PER_PAGE : (page + 1) * PER_PAGE]}
                    return self.send(200, json.dumps(r, default=str).encode())
            except re.error as e:
                return self.send(400, json.dumps({"error": f"Not a valid regex: {e}"}).encode())
            if u.path.startswith("/api/"):
                return self.send(404, b"{}")
            return super().do_GET()

        def imports_file(self, path):
            """Celeste's imports for the page, as HF would serve them: /api/tree/<dir> lists a directory, /files/<path> serves
            a file with byte ranges. Only under the imports root."""
            tree = path.startswith("/api/tree/")
            rel = path.removeprefix("/api/tree/" if tree else "/files/").removeprefix(IMPORTS + "/")
            root = corpus.imports.resolve()
            f = (root / rel).resolve()
            if not f.is_relative_to(root) or not f.exists():
                return self.send(404, b"{}")
            if tree:
                out = [
                    {"type": "directory" if x.is_dir() else "file", "path": f"{IMPORTS}/{x.relative_to(root)}", "size": x.stat().st_size}
                    for x in f.iterdir()
                ]
                return self.send(200, json.dumps(out).encode())
            size = f.stat().st_size
            m = re.match(r"bytes=(\d+)-(\d*)", self.headers.get("Range") or "")
            a, b = (int(m[1]), int(m[2] or size - 1)) if m else (0, size - 1)
            with open(f, "rb") as fh:
                fh.seek(a)
                body = fh.read(b - a + 1)
            self.send_response(206 if m else 200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Range")
            if m:
                self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Range")
            self.end_headers()

        def do_POST(self):
            if urlparse(self.path).path == "/api/reload":
                threading.Thread(target=corpus.refresh, daemon=True).start()
                return self.send(202, b'{"ok": true}')
            return self.send(404, b"{}")

        def log_message(self, format, *args):
            pass

    ThreadingHTTPServer.address_family = __import__("socket").AF_INET6 if ":" in host else __import__("socket").AF_INET
    httpd = ThreadingHTTPServer((host, port), H)
    print(f"serving {site} and /api on {host}:{port}", flush=True)
    httpd.serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(Path(__file__).parents[1] / "data"), help="the local mirror of the HF dataset")
    ap.add_argument("--imports", help="Celeste's imports (importers/celeste.py), default DATA/expview/imports/celeste")
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/expview-search"))
    ap.add_argument("--site", help="folder with only the two pages; omit to serve the API alone")
    ap.add_argument("--host", default="::")
    ap.add_argument("--port", type=int, default=8840)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--every", type=int, default=600, help="seconds between re-scans of the mirror")
    ap.add_argument("--build-only", action="store_true")
    a = ap.parse_args()
    corpus = Corpus(Path(a.data).resolve(), Path(a.imports or Path(a.data) / "expview/imports/celeste").resolve(), Path(a.cache), a.workers)
    corpus.refresh()
    if a.build_only:
        return

    def loop():
        while True:
            time.sleep(a.every)
            try:
                corpus.refresh()
            except Exception as e:  # noqa: BLE001
                print("refresh failed:", e, flush=True)

    threading.Thread(target=loop, daemon=True).start()
    site = Path(a.site) if a.site else Path(a.cache) / "_empty"
    site.mkdir(parents=True, exist_ok=True)
    serve(corpus, site, a.host, a.port)


if __name__ == "__main__":
    main()
