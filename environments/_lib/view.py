"""The env viewer on your machine, built from your working tree, for developing envs:

    uv run python -m environments view [ENV ...] [--port 8000] [--host 127.0.0.1] [--no-watch]

Builds the page and every env's spec from the local code (as the GitHub Pages workflow does from main) into
data/envs-view/ and serves only that folder: index.html, spec/ (from your code), src/ (copies of the source files the
specs name, for the Source tab) and envs -> data/envs (your local runs, rates and links). Nothing else in the repo is
reachable. With ENV names, only those envs are rebuilt at start (the rest keep their last build). While it runs, an env
whose source files change (env folder or _lib/) is rebuilt within a couple of seconds; reload the page to see it.
"""

import functools
import http.server
import json
import shutil
import socket
import threading
import time
from pathlib import Path

from .core import env_names
from .runfolder import LOCAL, REPO_ROOT
from .spec import build_site, source_files, source_hash

SITE = REPO_ROOT / "data" / "envs-view"


def _copy_sources(site: Path):
    files = set()
    for p in (site / "spec").glob("*/env.json"):
        files |= set(json.loads(p.read_text()).get("source_files") or [])
    shutil.rmtree(site / "src", ignore_errors=True)
    for f in sorted(files):
        dst = site / "src" / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO_ROOT / f, dst)


def build(envs: list[str] | None = None) -> Path:
    SITE.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO_ROOT / "environments/_lib/page.html", SITE / "index.html")
    build_site(SITE / "spec", envs)
    _copy_sources(SITE)
    link = SITE / "envs"
    if not link.exists():
        link.symlink_to(LOCAL, target_is_directory=True)
    return SITE


def _watch(every: float = 2.0):
    last = {e: source_hash(source_files(e)) for e in env_names()}
    page = (REPO_ROOT / "environments/_lib/page.html").stat().st_mtime
    while True:
        time.sleep(every)
        if (REPO_ROOT / "environments/_lib/page.html").stat().st_mtime != page:
            page = (REPO_ROOT / "environments/_lib/page.html").stat().st_mtime
            shutil.copy(REPO_ROOT / "environments/_lib/page.html", SITE / "index.html")
            print("page.html updated", flush=True)
        changed = [e for e in env_names() if last.get(e) != (h := source_hash(source_files(e))) and not last.update({e: h})]
        if changed:
            print("rebuilding", ", ".join(changed), flush=True)
            build(changed)


def cli(args: list[str]):
    opt = lambda k, d: args[args.index(k) + 1] if k in args else d
    port, host = int(opt("--port", "8000")), opt("--host", "127.0.0.1")
    envs = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] not in ("--port", "--host"))]
    build(envs or None)
    if "--no-watch" not in args:
        threading.Thread(target=_watch, daemon=True).start()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SITE))
    url = f"http://{host if ':' not in host else f'[{host}]'}:{port}/?base=envs/&src=src/"
    print(f"env viewer: {url}  (serving only {SITE})", flush=True)
    server = type("Server", (http.server.ThreadingHTTPServer,), {"address_family": socket.AF_INET6 if ":" in host else socket.AF_INET})
    server((host, port), handler).serve_forever()
