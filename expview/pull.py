# /// script
# requires-python = ">=3.12"
# dependencies = ["huggingface_hub>=0.25"]
# ///
"""Pull every run's expview files (and docent.json) from the sprint HF dataset into data/runs, so the server shows runs
other machines made. Never overwrites a file this machine wrote: a local file is only replaced when an earlier pull put
it there and it is unchanged since (data/expview/pulled.json records what each pull wrote).

    uv run --script expview/pull.py            # what expview-pull.timer runs on bramble
    uv run --script expview/pull.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile

REPO_ID = "Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens"
ROOT = Path(__file__).resolve().parent.parent / "data"
MANIFEST = ROOT / "expview/pulled.json"


def wanted(path: str) -> bool:
    parts = path.split("/")
    return parts[0] == "runs" and ("expview" in parts[:-1] or parts[-1] == "docent.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    seen = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    api = HfApi()
    n_new = n_upd = n_kept = 0
    for f in api.list_repo_tree(REPO_ID, repo_type="dataset", path_in_repo="runs", recursive=True):
        if not isinstance(f, RepoFile) or not wanted(f.path):
            continue
        oid = f.lfs.sha256 if f.lfs else f.blob_id
        local = ROOT / f.path
        prev = seen.get(f.path)
        if local.exists():
            st = local.stat()
            ours = prev and prev["size"] == st.st_size and prev["mtime"] == st.st_mtime_ns
            if not ours:
                n_kept += 1  # written on this machine (or edited since the pull): leave it
                continue
            if prev["oid"] == oid:
                continue
            n_upd += 1
        else:
            n_new += 1
        if a.dry_run:
            print(("update " if local.exists() else "new    ") + f.path)
            continue
        hf_hub_download(REPO_ID, f.path, repo_type="dataset", local_dir=ROOT)
        st = local.stat()
        seen[f.path] = {"oid": oid, "size": st.st_size, "mtime": st.st_mtime_ns}
    if not a.dry_run:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        tmp = MANIFEST.with_suffix(".tmp")
        tmp.write_text(json.dumps(seen, indent=0))
        os.replace(tmp, MANIFEST)
    print(f"{n_new} new, {n_upd} updated, {n_kept} local files left alone")


if __name__ == "__main__":
    main()
