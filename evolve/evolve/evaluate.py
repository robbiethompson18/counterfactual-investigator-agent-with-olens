"""Measure a node end to end: target batch and forecaster in parallel, then scores, summariser and rate audit.
A top-up is the same call on an already-measured node: it adds a fresh batch and re-audits. A newly audited node is
then compared for innocuousness against innocuous.K random eligible nodes (skipped once it has K comparisons)."""

import asyncio
import logging
import shutil

from . import innocuous, pool
from .audit import run_audit
from .forecast import run_forecaster, run_summariser
from .investigator import run_investigator
from .measure import batches, counts, run_batch
from .node import Node
from .store import Store

N_PER_ARM = 30
N_PICKS = 16
log = logging.getLogger(__name__)


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
    await run_investigator(node, picks, batch)  # transcript-only investigator: its U is blended into the surprise score
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
    await rate_innocuous(store, node)
    return s


async def rate_innocuous(store: Store, node: Node):
    """K innocuousness comparisons against eligible nodes (scored ones if fewer than K are eligible); never fatal."""
    others = [n for n in store.nodes() if n.id != node.id and n.scores().get("rank") and n.info.get("audit_score") is not None]
    el = pool.eligible(others)
    opp = el if len(el) >= innocuous.K else others
    try:
        rows = await innocuous.rate_new(node, opp)
    except Exception as e:  # a judge failure must not lose the measurement
        log.warning("innocuousness comparisons failed for %s", node.id, exc_info=True)
        store.event("innocuous_error", node=node.id, error=repr(e)[:300])
        return
    if rows:
        store.event("innocuous", node=node.id, comparisons=[{k: r[k] for k in ("id", "a", "b", "winner")} for r in rows])
