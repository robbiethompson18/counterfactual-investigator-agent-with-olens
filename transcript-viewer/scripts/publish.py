"""Share this machine's viewer data through the HF dataset, and rebuild the static site from everyone's.

Each machine owns its files on HF, so publishes never clash:
    weirdchat/reads/<machine>.jsonl      data/reads.jsonl      (lens reads)
    weirdchat/rollouts/<machine>.jsonl   data/claude.jsonl     (scripts/claude_rollouts.py, deleted 2026-10-26)
    weirdchat/stance/<machine>.jsonl     data/stance.jsonl     (stance labels)
    weirdchat/chive_judge/<machine>.jsonl  data/chive_judge.jsonl  (scripts/chive_judge.py)
    weirdchat/extras/<machine>.jsonl     data/extras.jsonl     (NLA + J++ reads, web/extras.py)
    weirdchat/site/...                   built from all of the above; the Cloudflare page reads it

Steps: push this machine's files, pull every other machine's into data/shared/weirdchat/ (web/build.py reads them
alongside its own), build one gzipped JSON per transcript (plus the pattern list), and upload the ones that changed.
CHIVE incidents ship without Agam's readout banks unless --banks: they aren't public, and they're ~4 GB of chunk files.

    .venv/bin/python scripts/publish.py                 # all of it
    .venv/bin/python scripts/publish.py --pull          # only pull other machines' data
    .venv/bin/python scripts/publish.py --preview DIR   # build the whole site into DIR (no HF writes), with an index.html
                                                        # that reads it: `python -m http.server -d DIR` to browse
    .venv/bin/python scripts/publish.py --banks         # also the CHIVE banks (bank/<pid>/<reader>/<chunk>)

The machine name is $WCO_MACHINE, else the short hostname.
"""

import argparse
import gc
import gzip
import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # the build forks after the tokenizer is loaded
import build
import chive
import extras

REPO = "Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens"
STAGE = ROOT / "data/site_stage"  # changed site files, laid out like the repo, for upload_large_folder
OUT = STAGE / "weirdchat/site"  # where put() writes; --preview points it elsewhere before the build forks
WORKERS = max(1, (os.cpu_count() or 2) - 2)
OLD = {}  # the last upload's manifest {path under weirdchat/site/: sha1}; build_site() fills it before forking
api = HfApi()


def push():
    ops = [CommitOperationAdd(f"weirdchat/{k}/{build.MACHINE}.jsonl", str(p)) for k, p in build.KINDS.items() if p.exists()]
    if ops:
        api.create_commit(REPO, ops, repo_type="dataset", commit_message=f"weirdchat: {build.MACHINE} data")
    print(f"pushed {[o.path_in_repo for o in ops]}", flush=True)


def pull():
    snapshot_download(
        REPO,
        repo_type="dataset",
        local_dir=build.SHARED.parent,  # weirdchat/<kind>/<machine>.jsonl lands under data/shared/
        allow_patterns=[f"weirdchat/{k}/*.jsonl" for k in build.KINDS],
        ignore_patterns=[f"weirdchat/*/{build.MACHINE}.jsonl"],
    )
    print(f"pulled into {build.SHARED}: {sorted(str(p.relative_to(build.SHARED)) for p in build.SHARED.glob('*/*.jsonl'))}", flush=True)


def put(rel, obj):
    """Gzip one site file; write it to OUT if it differs from the last upload. Returns {path: sha1}."""
    b = gzip.compress(json.dumps(obj, separators=(",", ":")).encode(), mtime=0)
    rel = f"{rel}.json.gz"
    sha = hashlib.sha1(b).hexdigest()
    if OLD.get(rel) != sha:
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b)
    return {rel: sha}


def build_overview():
    return put("overview", build.overview())


def build_pattern(i, banks):
    """One pattern's page and all its transcripts (run in a forked worker)."""
    S = build.S
    p = S["patterns"][i]
    out = put(f"pattern/{i}", build.pattern_detail(i))
    for tid in p["tids"]:
        t = build.transcript(tid)
        if p["ds"] == "chive" and banks:
            for r in t["readers"]:
                for c in range(-(-len(t["tokens"]) // chive.CHUNK) if r["layers"] else 0):
                    out |= put(f"bank/{tid}/{r['key']}/{c}", chive.bank_chunk(tid, r["key"], c))
        elif p["ds"] == "chive":  # the scored bank reads hold bank text too
            t.update(readers=[dict(r, layers=[]) for r in t["readers"]], bank_reads=[])
        out |= put(f"t/{tid}", t)
        for c in sorted({pos // extras.CHUNK for pos in extras.of(tid)}):
            out |= put(f"x/{tid}/{c}", extras.chunk(tid, c))
        S["rendered"].pop(tid)  # 25k rendered transcripts don't fit comfortably in memory
    return out


def build_site(banks, preview=None, force=False):
    """Write everything the page fetches (gzipped JSON) to OUT, laid out like weirdchat/site/ in the repo; returns the
    manifest {path: sha1}. Publishing writes only what changed since the last upload, to STAGE; --preview writes it all
    to that directory, plus an index.html that reads from it.

    Rendering is CPU-bound (tokenising, per-token decoding, gzip), so patterns are built in WORKERS forked processes that
    share the loaded reads copy-on-write."""
    global OUT
    if preview:
        OUT = Path(preview).resolve()
    else:
        check_current(force)  # here too, so calling build_site() directly can't skip it
        try:
            OLD.update(json.loads(Path(hf_hub_download(REPO, "weirdchat/site/manifest.json", repo_type="dataset")).read_text()))
        except EntryNotFoundError:
            pass
    build.init()
    S = build.S
    print(f"{len(S['trs'])} transcripts, {len(S['reads'])} reads", flush=True)
    shutil.rmtree(OUT, ignore_errors=True)
    manifest = put("patterns", {"patterns": [build.pattern_summary(p) for p in S["patterns"]], "layers": build.LAYERS})
    # biggest patterns first, so no worker is left with a long one at the end
    order = sorted(range(len(S["patterns"])), key=lambda i: -len(S["patterns"][i]["tids"]))
    gc.freeze()  # keep the collector from writing to (and so copying) every inherited object in each worker
    t0 = time.time()
    with multiprocessing.get_context("fork").Pool(WORKERS) as pool:
        overview = pool.apply_async(build_overview)
        for k, m in enumerate(pool.imap_unordered(partial(build_pattern, banks=banks), order), 1):
            manifest |= m
            if k % 50 == 0 or k == len(order):
                print(f"{k}/{len(order)} patterns, {time.time() - t0:.0f}s", flush=True)
        manifest |= overview.get()
    gc.unfreeze()
    if preview:
        html = (ROOT / "web/index.html").read_text().replace("</head>", "<script>window.WCO_STATIC = './';</script></head>", 1)
        (OUT / "index.html").write_text(html)
        print(f"site in {OUT}: python -m http.server -d {OUT}", flush=True)
    return manifest


def upload_site(manifest):
    n = sum(1 for f in STAGE.rglob("*") if f.is_file()) if STAGE.exists() else 0
    print(f"{n}/{len(manifest)} site files changed", flush=True)
    if n:
        api.upload_large_folder(REPO, folder_path=str(STAGE), repo_type="dataset")
        shutil.rmtree(STAGE)
    # last, so an interrupted upload gets redone next time
    api.upload_file(
        path_or_fileobj=json.dumps(manifest).encode(), path_in_repo="weirdchat/site/manifest.json", repo_id=REPO, repo_type="dataset"
    )


def check_current(force=False):
    """The site's JSON comes from this checkout's web/ code, but the page from whatever deploy_site.sh last shipped (main):
    publishing from other code breaks the page, and flips every file so the whole site re-uploads (~15 min). Code
    behind main is refused even with force: its pattern list drops whatever main added since (a stale worktree's
    build wiped the confession follow-ups off the site on 2026-10-20)."""
    git = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, check=False)
    git("fetch", "-q", "origin", "main")
    if git("merge-base", "--is-ancestor", "origin/main", "HEAD").returncode:
        sys.exit("this checkout is behind origin/main: pull main first (--force doesn't override this)")
    if not force and git("diff", "--quiet", "origin/main", "--", "web", "scripts/publish.py").returncode:
        sys.exit("transcript-viewer/web or publish.py differs from origin/main: pull main (or merge your viewer PR) first, or --force")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true", help="only pull other machines' data")
    ap.add_argument("--preview", metavar="DIR", help="build the whole site into DIR for a local http.server; no HF writes")
    ap.add_argument("--banks", action="store_true", help="also publish Agam's CHIVE readout banks (not public: ask first)")
    ap.add_argument("--force", action="store_true", help="publish even if the viewer code differs from origin/main")
    args = ap.parse_args()
    if args.preview:
        return build_site(args.banks, preview=args.preview)
    if not args.pull:
        check_current(args.force)
        push()
    pull()
    if not args.pull:
        upload_site(build_site(args.banks, force=args.force))


if __name__ == "__main__":
    main()
