"""Re-grade a node's stored runs with its current grader.py (after a grader fix) and rescore it.
.venv/bin/python scripts/regrade.py v1 <node id> [...]"""

import asyncio
import json
import sys

from evolve.grading import grade_records, verdict
from evolve.measure import batches
from evolve.node import ARMS
from evolve.store import Store

from evolve import pool


async def regrade(store, node):
    image = node.arm("baseline").episode.get("image", "evalcue-sandbox")
    changed = 0
    for b in batches(node):
        for a in ARMS:
            p = node.dir / "runs" / b / f"{a}.jsonl"
            rows = [json.loads(line) for line in p.read_text().splitlines()]
            ok = [r for r in rows if "error" not in r["record"]]
            vs = iter(await grade_records(node.grader_src, [r["record"] for r in ok], image=image))
            for r in ok:
                v = next(vs)
                new = verdict(v)
                changed += new != r["verdict"]
                r["verdict"], r["grader_error"] = new, v.get("error") if isinstance(v, dict) else None
            p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    pool.rescore(node)
    store.event("regraded", node=node.id, changed=changed)
    return changed


async def main(run, ids):
    store = Store(run)
    for i in ids:
        print(i, await regrade(store, store.node(i)))
    store.reindex()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2:]))
