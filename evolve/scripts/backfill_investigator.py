"""Run the transcript-only investigator on every measured node that doesn't have it for its latest batch yet.
.venv/bin/python scripts/backfill_investigator.py v1 [--n 16] [--conc 6]"""

import argparse
import asyncio

from evolve.investigator import run_investigator
from evolve.measure import batches
from evolve.store import Store


async def main(run, n, conc):
    store = Store(run)
    sem = asyncio.Semaphore(conc)

    async def one(node):
        b = batches(node)[-1]
        if (node.dir / "investigator" / f"{b}.jsonl").exists():
            return
        async with sem:
            res = await run_investigator(node, n, b)
        store.event("investigated", node=node.id, batch=b, investigator=res)

    await asyncio.gather(*(one(x) for x in store.nodes() if batches(x)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--conc", type=int, default=6)
    a = ap.parse_args()
    asyncio.run(main(a.run, a.n, a.conc))
