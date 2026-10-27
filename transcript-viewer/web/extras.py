"""NLA (ceselder L42) and J++ lens reads for the viewer, from data/extras.jsonl.

One record per (transcript, token, kind):
  nla: {text, cos, fve}  the AV's explanation of the L42 activation and the AR's reconstruction of it
  jpp: {layers, top}     top-k next-token predictions of the J++ lens at every block 0..62
"""

import json
from pathlib import Path

CACHE = Path(__file__).resolve().parents[1] / "data/extras.jsonl"  # written by scripts/read_extras.py (deleted 2026-10-26, at 7296169)
E = {}  # tid -> {pos: {kind: record}}
CHUNK = 64  # tokens per file on the read-only site


def load(paths):
    """Every machine's shard (build.shards("extras")), this machine's last so its records win."""
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
