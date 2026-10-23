"""The pool on disk: data/<run>/nodes/<id>/, an append-only events log and a node index, all plain JSON so a
dashboard can follow the loop live by tailing events.jsonl and re-reading index.json.

    data/<run>/
      config.json      the run's settings
      events.jsonl     one JSON object per line: {"t", "event", "node"?, ...}
      index.json       {node_id: {parent, refs, kind, round, stage, scores summary}}, rewritten after every change
      nodes/<id>/      see node.py
"""

import json
import os
import time
import uuid
from pathlib import Path

from .node import Node

ROOT = Path(os.environ.get("EVOLVE_DATA", Path(__file__).resolve().parent.parent / "data"))


class Store:
    def __init__(self, run: str):
        self.dir = ROOT / run
        (self.dir / "nodes").mkdir(parents=True, exist_ok=True)

    def node(self, nid: str) -> Node:
        return Node(self.dir / "nodes" / nid)

    def nodes(self) -> list[Node]:
        return [Node(p) for p in sorted((self.dir / "nodes").iterdir()) if (p / "node.json").exists()]

    def new_id(self, prefix: str = "n") -> str:
        return f"{prefix}{time.strftime('%m%d%H%M%S', time.gmtime())}-{uuid.uuid4().hex[:4]}"

    def event(self, event: str, **kw):
        with open(self.dir / "events.jsonl", "a") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, **kw}) + "\n")

    def reindex(self):
        idx = {}
        for n in self.nodes():
            i, s = n.info, n.scores()
            idx[n.id] = {k: i.get(k) for k in ("parent", "refs", "kind", "round", "stage", "title")} | {
                "behaviour": (n.dir / "behaviour.txt").read_text().strip() if (n.dir / "behaviour.txt").exists() else None,
                "scores": s.get("rank"),
                "rates": s.get("pooled_rates"),
                "audit": i.get("audit_score"),
            }
        tmp = self.dir / "index.json.tmp"
        tmp.write_text(json.dumps(idx, indent=1) + "\n")
        tmp.replace(self.dir / "index.json")
