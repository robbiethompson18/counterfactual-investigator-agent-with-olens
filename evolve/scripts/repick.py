"""Re-run the Forecaster (and Summariser) on every measured node, keeping the old trials in picker_superseded/.

Used once on 2026-10-23: the first seed batch's forecaster prompts had arm names ("+++ real/...") in the diff headers.
    .venv/bin/python scripts/repick.py v1 [node ids]
"""

import asyncio
import shutil
import sys

from evolve.forecast import run_forecaster, run_summariser
from evolve.measure import batches, counts
from evolve.store import Store

from evolve import pool


async def repick(store, node):
    old = node.dir / "picker"
    if old.exists():
        dest = node.dir / "picker_superseded"
        shutil.rmtree(dest, ignore_errors=True)
        old.rename(dest)
    for b in batches(node):
        await run_forecaster(node, 16, b)
    pool.rescore(node)
    await run_summariser(node, {a: v[:2] for a, v in counts(node).items()})
    store.event("repicked", node=node.id, picker=node.scores()["picker"])


async def main(run, ids):
    store = Store(run)
    nodes = [n for n in store.nodes() if batches(n) and (not ids or n.id in ids)]
    await asyncio.gather(*(repick(store, n) for n in nodes))
    store.reindex()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2:]))
