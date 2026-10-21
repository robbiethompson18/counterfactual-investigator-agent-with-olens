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
CHUNK = 64  # tokens per file on the read-only site


def load(paths):
    """Every machine's shard (server.shards("extras")), this machine's last so its records win."""
    for p in paths:
        for line in p.open():
            r = json.loads(line)
            E.setdefault(r["tid"], {}).setdefault(r["pos"], {})[r["kind"]] = r


def of(tid):
    return E.get(tid, {})


def chunk(tid, c):
    """{pos: {kind: record}} for tokens c*CHUNK .. (c+1)*CHUNK-1, as the read-only site serves them (scripts/publish.py):
    no bookkeeping fields, J++ probabilities to 4 places."""
    out = {}
    for p, recs in E.get(tid, {}).items():
        if p // CHUNK == c:
            out[p] = {k: {f: v for f, v in r.items() if f not in ("tid", "pos", "kind", "t")} for k, r in recs.items()}
            if "jpp" in out[p]:
                out[p]["jpp"]["top"] = [[[t, round(q, 4)] for t, q in row] for row in out[p]["jpp"]["top"]]
    return out


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
                    recs.append({"tid": tid, "pos": p, "kind": "jpp", "t": t, "layers": res["jpp_layers"], "top": top})
        with LOCK, CACHE.open("a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
                E.setdefault(tid, {}).setdefault(r["pos"], {})[r["kind"]] = r
    return {p: E.get(tid, {}).get(p, {}) for p in positions}
