"""CHIVE incidents from the public HF dataset agu18dec/chive-forensics (docs/chive-data-sources.md).

An incident is its `incidents` row (messages, completion, behaviour, k, n, ...) plus `token_ids`, `pieces` and `turn_of`
from `tokens`, with the whole `ground_truth` row under "ground_truth". All three configs are joined on prompt_id.
"""

import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

HF_REPO = "agu18dec/chive-forensics"


def load_incidents(ids: list[str]) -> dict[str, dict[str, Any]]:
    rows = {}
    for config in ("incidents", "tokens", "ground_truth"):
        path = hf_hub_download(HF_REPO, f"data/{config}.jsonl", repo_type="dataset")
        rows[config] = {r["prompt_id"]: r for r in map(json.loads, Path(path).read_text().splitlines())}
    return {pid: {**rows["incidents"][pid], **rows["tokens"][pid], "ground_truth": rows["ground_truth"][pid]} for pid in ids}
