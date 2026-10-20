"""NLA (ceselder L42) and J++ lens reads for the viewer, cached in data/extras.jsonl.

One record per (transcript, token, kind):
  nla: {text, cos, fve}  the AV's explanation of the L42 activation and the AR's reconstruction of it
  jpp: {layers, top}     top-k next-token predictions of the J++ lens at every block 0..62
"""

import json
import threading
import time
from pathlib import Path

CACHE = Path(__file__).resolve().parents[1] / "data/extras.jsonl"
LOCK = threading.Lock()
E = {}  # tid -> {pos: {kind: record}}


def load():
    if CACHE.exists():
        for line in CACHE.open():
            r = json.loads(line)
            E.setdefault(r["tid"], {}).setdefault(r["pos"], {})[r["kind"]] = r


def of(tid):
    return E.get(tid, {})


def read(pod, tok, tid, ids, positions, kinds=("nla", "jpp")):
    """Read the positions not cached yet on the pod; return {pos: {kind: record}} for all of them."""
    have = E.get(tid, {})
    todo = {k: [p for p in positions if k not in have.get(p, {})] for k in kinds}
    todo = {k: v for k, v in todo.items() if v}
    if todo:
        if not (pod and pod.ready()):
            raise RuntimeError("the GPU pod isn't running: start it from the header (cached reads still show)")
        need = sorted({p for v in todo.values() for p in v})
        recs = []
        for i in range(0, len(need), 64):
            chunk = need[i : i + 64]
            res = pod.call("/extras", {"ids": ids, "positions": chunk, "nla": "nla" in todo, "jpp": "jpp" in todo, "k": 8})
            t = time.time()
            for j, p in enumerate(chunk):
                if "nla" in res and p in todo.get("nla", []):
                    recs.append(dict(tid=tid, pos=p, kind="nla", t=t, **res["nla"][j]))
                if "jpp" in res and p in todo.get("jpp", []):
                    top = [[[tok.decode([x]), q] for x, q in row] for row in res["jpp"][str(p)]]
                    recs.append(dict(tid=tid, pos=p, kind="jpp", t=t, layers=res["jpp_layers"], top=top))
        with LOCK:
            with CACHE.open("a") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
                    E.setdefault(tid, {}).setdefault(r["pos"], {})[r["kind"]] = r
    return {p: E.get(tid, {}).get(p, {}) for p in positions}
