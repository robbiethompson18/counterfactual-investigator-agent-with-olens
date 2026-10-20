"""Mirror every per-token readout bank in Agam's CHIVE cases viewer (O-lens, multi-layer NLA, J-lens,
L42 NLA) for the given incidents, as served: <out>/<pid>/<reader>_L{L}.json.gz, each a list indexed by
token position. Banks an incident lacks (404) are skipped. Resumable.

    uv run python scripts/mirror_agam_readers.py splits/all300.txt data/banks/agam
"""

import gzip
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

VIEWER = "https://safety-sahan-2-agam--chive-cases-web.modal.run/data/r"
READERS = {
    "olens_bank": range(20, 61, 4),
    "mlnla": range(20, 61, 4),
    "jlens_all": range(63),
    "nla_k4": [42],
}

ids_file, out = Path(sys.argv[1]), Path(sys.argv[2])
ids = [line.strip() for line in ids_file.read_text().splitlines() if line.strip()]


def fetch(task: tuple[str, str, int]) -> str:
    pid, reader, layer = task
    dest = out / pid / f"{reader}_L{layer}.json.gz"
    if dest.exists():
        return ""
    for attempt in range(5):  # Modal's DNS and gateway blip now and then
        try:
            resp = requests.get(f"{VIEWER}/{pid}/{reader}_L{layer}.json", timeout=300)
            if resp.status_code < 500:
                break
        except requests.ConnectionError:
            if attempt == 4:
                raise
        time.sleep(10 * (attempt + 1))
    if resp.status_code == 404:
        return ""
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(resp.content))
    tmp.rename(dest)
    return f"{pid} {reader} L{layer} {dest.stat().st_size / 1e6:.2f}MB"


tasks = [(pid, reader, layer) for pid in ids for reader, layers in READERS.items() for layer in layers]
with ThreadPoolExecutor(32) as pool:
    for line in pool.map(fetch, tasks):
        if line:
            print(line, flush=True)
