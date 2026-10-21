"""NLA (ceselder L42) + J++ reads at every token of some transcripts, against pod/server.py (with --extras on, i.e. an
H200) on 127.0.0.1:PORT, e.g. a RunPod pod over an ssh tunnel. Goes through the viewer's own code (web/extras.py) into
data/extras.jsonl, skipping tokens already read; no viewer running, so its precompute doesn't use the pod too.

    .venv/bin/python scripts/read_extras.py 8802                  # docs/favorite-transcripts.md
    .venv/bin/python scripts/read_extras.py 8802 TID [TID ...]
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "web"))
import extras
import server
from local import Local

# docs/favorite-transcripts.md
FAVORITES = [
    "644ba62154d0eee979268fafb9e22d7b986e2b5250e945d7b8fb6ad76abf768d63/t044",  # hospital / claims to call 911
    "641e2ba3216942ec66f7e4fb851427d9c908edc6cff3dd6bc2421f2b3b6bffd77d/t054",  # unsolicited sexual advances
    "64bb7e454c2534db8fccd094a80416d8f7353753d3a9a9931a203fa6359b469c3c/t053",  # recommends drunk driving
]
STEP = 256  # tokens per progress line (extras.read sends 64 at a time)


def main():
    port, tids = sys.argv[1], sys.argv[2:] or FAVORITES
    server.init()
    extras.load(server.shards("extras"))
    pod = Local(f"http://127.0.0.1:{port}")
    while not pod.ready():
        print("waiting for the lens server", flush=True)
        time.sleep(30)
    t0 = time.time()
    for tid in tids:
        ids = server.render(tid)["ids"]
        for a in range(0, len(ids), STEP):
            extras.read(pod, server.S["tok"], tid, ids, list(range(a, min(a + STEP, len(ids)))))
            print(f"{tid}: {min(a + STEP, len(ids))}/{len(ids)} tokens, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
