"""Download Agam's OLens readout banks from his CHIVE cases viewer into the layout
Agam's harness (vendor/detective-joracle, reference-only) reads: <out>/<prompt_id>.json.gz with token_ids, k and
readouts[layer][pos] -> [samples].

The viewer serves one file per (incident, layer): a list indexed by token position, each entry the
k samples at that cell. Token ids come from the HF dataset so the reader's tokenization check passes.

    uv run python scripts/fetch_olens_banks.py splits/smoke25.txt data/banks/olens_rl
"""

import gzip
import json
import sys
from pathlib import Path

import requests
from chive_data import load_incidents  # sibling module; the script dir is on sys.path

VIEWER = "https://safety-sahan-2-agam--chive-cases-web.modal.run/data/r"
LAYERS = range(20, 61, 4)

ids_file, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
ids = ids_file.read_text().split()
token_ids = {pid: inc["token_ids"] for pid, inc in load_incidents(ids).items()}

for pid in ids:
    dest = out / f"{pid}.json.gz"
    if dest.exists():
        continue
    readouts = {}
    for layer in LAYERS:
        cells = requests.get(f"{VIEWER}/{pid}/olens_bank_L{layer}.json", timeout=120).json()
        if len(cells) != len(token_ids[pid]):
            raise SystemExit(f"{pid} L{layer}: {len(cells)} cells but {len(token_ids[pid])} tokens")
        readouts[str(layer)] = {str(pos): samples for pos, samples in enumerate(cells) if samples}
    record = {"prompt_id": pid, "token_ids": token_ids[pid], "k": 4, "readouts": readouts}
    with gzip.open(dest, "wt") as f:
        json.dump(record, f, ensure_ascii=False)
    print(pid, f"{dest.stat().st_size / 1e6:.1f} MB", flush=True)
