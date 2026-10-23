"""Measure a node end to end: target batch and forecaster in parallel, then scores, summariser and rate audit.
A top-up is the same call on an already-measured node: it adds a fresh batch and re-audits."""

import asyncio
import shutil

from . import pool
from .audit import run_audit
from .forecast import run_forecaster, run_summariser
from .measure import batches, counts, run_batch
from .node import Node
from .store import Store

N_PER_ARM = 30
N_PICKS = 16


def drop_partial(node: Node):
    """Remove batches an interrupted process left half-written (and their forecaster trials)."""
    for b in batches(node):
        d = node.dir / "runs" / b
        if not all((d / f"{a}.jsonl").exists() for a in ("baseline", "real", "null")):
            shutil.rmtree(d)
            (node.dir / "picker" / f"{b}.jsonl").unlink(missing_ok=True)


async def evaluate(store: Store, node: Node, n=N_PER_ARM, picks=N_PICKS) -> dict:
    drop_partial(node)
    batch = f"b{len(batches(node))}"
    store.event("measure_start", node=node.id, batch=batch, n=n, picks=picks)
    node.update(stage="measuring")
    store.reindex()
    await asyncio.gather(run_batch(node, n, batch), run_forecaster(node, picks, batch))
    s = pool.rescore(node)
    store.event(
        "measured",
        node=node.id,
        batch=batch,
        rates=s["latest_rates"],
        picker=s["picker"]["latest"],  # event key kept for readers of older events
        rank={k: s["rank"][k] for k in ("L", "U", "U_rev", "N", "P_pos")},
    )
    pooled = {a: v[:2] for a, v in counts(node).items()}
    await run_summariser(node, pooled)
    audit = await run_audit(node, batch)
    node.update(stage="scored")
    store.event("audited", node=node.id, batch=batch, score=audit["score"])
    store.reindex()
    return s
