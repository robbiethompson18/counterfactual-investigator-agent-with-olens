# /// script
# dependencies = ["huggingface_hub"]
# ///
"""Build the WorkspaceBench item banks from the pinned submodule and put them on HF (banks are data: never commit them).

    git submodule update --init vendor/workspace-bench
    uv run --script wsb/build_banks.py            # writes data/banks/wsb/<family>.json
    uv run --script wsb/build_banks.py --upload   # ... and uploads them to banks/wsb/ in the sprint HF dataset

Each bank is their `evals/<family>/items.json` verbatim under a `source` header (repo, path, commit, license). Reading
data files from vendor/workspace-bench is fine; importing or running its code is not (vendor/CLAUDE.md)."""

import argparse
import json
import subprocess
from pathlib import Path

from huggingface_hub import HfApi

REPO = "Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens"  # runs.REPO (this script can't import the package)
WSB = Path("vendor/workspace-bench")
OUT = Path("data/banks/wsb")
NOTES = {
    "user_modeling": "100 items copied verbatim (family, gate, read, items). Originally Transluce's SelfDescribe / "
    "SynthSysPre rows (Choi et al. 2025).",
}


def build(family: str, commit: str) -> dict:
    path = f"evals/{family}/items.json"
    source = {
        "repo": "https://github.com/camilablank/workspace-bench",
        "path": path,
        "commit": commit,
        "author": "Camila Blank",
        "license": "MIT",
        "note": NOTES[family],
    }
    return {"source": source, **json.loads((WSB / path).read_text())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--upload", action="store_true", help="also upload data/banks/wsb/ to banks/wsb/ on HF")
    args = ap.parse_args()
    commit = subprocess.run(["git", "-C", str(WSB), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    OUT.mkdir(parents=True, exist_ok=True)
    for family in NOTES:
        bank = build(family, commit)
        (OUT / f"{family}.json").write_text(json.dumps(bank, indent=1, ensure_ascii=False) + "\n")
        print(f"{OUT}/{family}.json: {len(bank['items'])} items from {WSB}@{commit[:7]}")
    if args.upload:
        HfApi().upload_folder(
            repo_id=REPO,
            repo_type="dataset",
            folder_path=str(OUT),
            path_in_repo="banks/wsb",
            commit_message=f"banks/wsb: WorkspaceBench item banks from workspace-bench@{commit[:7]}",
        )
        print(f"uploaded -> https://huggingface.co/datasets/{REPO}/tree/main/banks/wsb")


if __name__ == "__main__":
    main()
