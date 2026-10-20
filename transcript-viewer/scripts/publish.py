"""Share this machine's viewer data through the HF dataset, and rebuild the read-only site from everyone's.

Each machine owns its files on HF, so publishes never clash:
    weirdchat/reads/<machine>.jsonl      data/reads.jsonl      (lens reads)
    weirdchat/rollouts/<machine>.jsonl   data/claude.jsonl     (scripts/claude_rollouts.py)
    weirdchat/stance/<machine>.jsonl     data/stance.jsonl     (stance labels)
    weirdchat/chive_judge/<machine>.jsonl  data/chive_judge.jsonl  (scripts/chive_judge.py)
    weirdchat/site/...                   built from all of the above; the Cloudflare page reads it

Steps: push this machine's files, pull every other machine's into data/shared/weirdchat/ (web/server.py reads them
alongside its own), build one gzipped JSON per transcript (plus the pattern list), and upload the ones that changed.
CHIVE incidents ship without Agam's readout banks unless --banks: they aren't public, and they're ~4 GB of chunk files.

    .venv/bin/python scripts/publish.py                 # all of it
    .venv/bin/python scripts/publish.py --pull          # only pull other machines' data, to browse it locally
    .venv/bin/python scripts/publish.py --seed-from http://bramble:8803 --as bramble   # copy a viewer's reads over HTTP
    .venv/bin/python scripts/publish.py --banks         # also the CHIVE banks (bank/<pid>/<reader>/<chunk>)

The machine name is $WCO_MACHINE, else the short hostname.
"""

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))
import chive
import server

REPO = "Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens"
STAGE = ROOT / "data/site_stage"  # changed site files, laid out like the repo, for upload_large_folder
api = HfApi()


def push():
    ops = [CommitOperationAdd(f"weirdchat/{k}/{server.MACHINE}.jsonl", str(p)) for k, p in server.KINDS.items() if p.exists()]
    if ops:
        api.create_commit(REPO, ops, repo_type="dataset", commit_message=f"weirdchat: {server.MACHINE} data")
    print(f"pushed {[o.path_in_repo for o in ops]}", flush=True)


def pull():
    snapshot_download(
        REPO,
        repo_type="dataset",
        local_dir=server.SHARED.parent.parent,
        allow_patterns=[f"weirdchat/{k}/*.jsonl" for k in server.KINDS],
        ignore_patterns=[f"weirdchat/*/{server.MACHINE}.jsonl"],
    )
    print(f"pulled into {server.SHARED}: {sorted(str(p.relative_to(server.SHARED)) for p in server.SHARED.glob('*/*.jsonl'))}", flush=True)


def seed(url, name):
    """Copy every cached read from a running viewer (e.g. one we can't ssh into) into its shard, and push it."""
    get = lambda path: json.load(urllib.request.urlopen(url + path, timeout=300))
    server.load()  # the same WeirdChat copy, so the same transcript ids
    tids = [tid for p in server.S["patterns"] for tid in p["tids"]]
    out = server.SHARED / "reads" / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    t0, n_reads = time.time(), 0
    with out.open("w") as f, ThreadPoolExecutor(10) as ex:  # ~1 s/transcript, mostly transfer; the viewer also serves its fleet
        for k, reads in enumerate(ex.map(lambda t: get(f"/api/transcript?tid={quote(t)}")["reads"], tids)):
            f.write("".join(json.dumps(r) + "\n" for r in reads))
            n_reads += len(reads)
            if k % 1000 == 0:
                print(f"{k}/{len(tids)} transcripts, {n_reads} reads, {time.time() - t0:.0f}s", flush=True)
    api.upload_file(path_or_fileobj=str(out), path_in_repo=f"weirdchat/reads/{name}.jsonl", repo_id=REPO, repo_type="dataset")
    print(f"seeded {n_reads} reads from {url} as {name}", flush=True)


def build(banks):
    """Write everything the page fetches that changed since the last upload (gzipped JSON, the same shapes as the server's
    API) to STAGE, laid out like the repo; returns the new manifest {path under weirdchat/site/: sha1}."""
    server.init()
    print(f"{len(server.S['trs'])} transcripts, {len(server.S['reads'])} reads", flush=True)
    try:
        old = json.loads(Path(hf_hub_download(REPO, "weirdchat/site/manifest.json", repo_type="dataset")).read_text())
    except EntryNotFoundError:
        old = {}
    shutil.rmtree(STAGE, ignore_errors=True)
    manifest = {}

    def put(rel, obj):
        b = gzip.compress(json.dumps(obj, separators=(",", ":")).encode(), mtime=0)
        rel = f"{rel}.json.gz"
        manifest[rel] = hashlib.sha1(b).hexdigest()
        if old.get(rel) != manifest[rel]:
            dst = STAGE / "weirdchat/site" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b)

    S = server.S
    put(
        "patterns",
        {"patterns": [server.pattern_summary(p) for p in S["patterns"]], "layers": server.LAYERS, "pre_layers": server.PRE_LAYERS},
    )
    put("overview", server.overview())
    for p in S["patterns"]:
        put(f"pattern/{p['idx']}", server.pattern_detail(p["idx"]))
        for tid in p["tids"]:
            t = dict(server.transcript(tid), pattern=p["idx"])  # pattern: answers /api/locate
            if p["ds"] == "chive" and banks:
                for r in t["readers"]:
                    for c in range(-(-len(t["tokens"]) // chive.CHUNK) if r["layers"] else 0):
                        put(f"bank/{tid}/{r['key']}/{c}", chive.bank_chunk(tid, r["key"], c))
            elif p["ds"] == "chive":  # the scored bank reads hold bank text too
                t.update(readers=[dict(r, layers=[]) for r in t["readers"]], bank_reads=[])
            put(f"t/{tid}", t)
            S["rendered"].pop(tid)  # 25k rendered transcripts don't fit comfortably in memory
        print(f"pattern {p['idx'] + 1}/{len(S['patterns'])}", flush=True)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true", help="only pull other machines' data")
    ap.add_argument("--seed-from", help="a viewer URL to copy reads from (then also --as)")
    ap.add_argument("--as", dest="name", help="the machine name for --seed-from")
    ap.add_argument("--banks", action="store_true", help="also publish Agam's CHIVE readout banks (not public: ask first)")
    args = ap.parse_args()
    if args.seed_from:
        if not args.name:
            ap.error("--seed-from needs --as NAME")
        return seed(args.seed_from.rstrip("/"), args.name)
    if not args.pull:
        push()
    pull()
    if not args.pull:
        upload_site(build(args.banks))


if __name__ == "__main__":
    main()
