"""WorkspaceBench item banks: data, so they live in the sprint HF dataset (`banks/wsb/<family>.json`), never in git.
`build_banks.py` makes them from the pinned `vendor/workspace-bench` submodule; this module only reads them."""

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from counterfactual_investigator_agent_with_olens.runs import REPO


def load(family: str) -> dict:
    """data/banks/wsb/<family>.json, downloaded from the HF dataset on first use (so any checkout works)."""
    path = Path("data/banks/wsb") / f"{family}.json"
    if not path.exists():
        hf_hub_download(REPO, f"banks/wsb/{family}.json", repo_type="dataset", local_dir="data")
    return json.loads(path.read_text())
