"""The evolution loop.

A fixed number of workers each run one job at a time, drawn from the pool as it stands when the job starts, so
the GPUs measuring one child stay busy while other workers' agents think. A job is:
  - a top-up, if an eligible node is due one (fresh batch + forecaster trials, then rescore and re-audit);
  - with probability 0.1, a creation: the Creator invents a node from 6 examples, the Novelty checker admits it as a
    new root or rejects it as a duplicate;
  - otherwise a mutation: a reference and two secondaries are drawn (surprise score U or U' by coin flip), the
    Evolver edits a copy of the reference in /rw/child and submits, and the child is measured and scored.
Everything is logged to events.jsonl and index.json as it happens (see store.py).
"""

import asyncio
import json
import os
import random
import shutil
from types import SimpleNamespace

from . import job_queue, pool, workspace
from .agent import run_workspace
from .evaluate import evaluate
from .novelty import check_novelty
from .store import Store

SLOTS = 4  # job starts per epoch (front counters, top-up timing)
P_CREATOR = 0.1
N_EXAMPLES = 6
AGENT_MAX_COST = 3.0
BACKEND = "queue"  # "queue": agent jobs go to data/<run>/agent_jobs for an external manager; "api": Sonnet via OpenRouter


AGENT_SLOTS = asyncio.Semaphore(int(os.environ.get("EVOLVE_AGENT_SLOTS", "2")))  # agent jobs in flight


async def run_agent_job(ws) -> dict:
    async with AGENT_SLOTS:
        if BACKEND == "api":
            return await run_workspace(ws, log=ws.root / "session.json", max_cost=AGENT_MAX_COST)
        root = ws.root if (ws.root / "job.json").exists() else job_queue.post(ws)  # already posted: a resumed job
        return await job_queue.wait(root)


def new_workspace(store: Store, make, *args):
    base = store.dir / "agent_jobs"
    base.mkdir(exist_ok=True)
    return make(*args, base=base if BACKEND == "queue" else None)


def keep_session(ws, node):
    """Copy the agent's transcript (API) or job instructions + manager notes (queue) into the node."""
    names = ("session.json", "INSTRUCTIONS.md", "transcript.md", "transcript.json", "transcript.jsonl", "manager.json", "probes.jsonl")
    for name in (*names, "claimed.json", "result.json"):
        if (ws.root / name).exists():
            shutil.copy2(ws.root / name, node.dir / f"agent_{name}")


def _admit(store: Store, ws, prefix: str, **info):
    nid = store.new_id(prefix)
    node = store.node(nid)
    shutil.copytree(ws.child, node.dir)
    node.update(id=nid, stage="submitted", **info)
    return node


def _plan(ws, **plan):
    """What to do with this workspace's result; lets a restarted loop finish jobs posted by an earlier one."""
    (ws.root / "loop.json").write_text(json.dumps({**plan, "closed": False}) + "\n")


def _close(ws):
    p = ws.root / "loop.json"
    if p.exists():
        p.write_text(json.dumps({**json.loads(p.read_text()), "closed": True}) + "\n")
    if BACKEND == "api":
        ws.cleanup()  # queue jobs stay on disk as provenance


async def _finish(store: Store, ws, plan: dict, res: dict):
    """After the agent: admit the node (creations pass the novelty check first), then measure it."""
    rnd = plan["round"]
    node = None
    try:
        if not res["submitted"]:
            store.event(f"{plan['op']}_failed", round=rnd, reference=plan.get("parent"), errors=res["errors"], cost=res["cost"])
            return
        if plan["op"] == "creation":
            nov = await check_novelty(ws.child, store)
            store.event("novelty", round=rnd, **{k: nov.get(k) for k in ("new", "duplicate_of", "reason")})
            if not nov.get("new"):
                return
            node = _admit(store, ws, "c", parent=None, refs=plan["refs"], kind="creator", round=rnd, job=ws.root.name)
            (node.dir / "novelty.json").write_text(json.dumps(nov, indent=1) + "\n")
        else:
            node = _admit(store, ws, "m", parent=plan["parent"], refs=plan["refs"], kind="child", round=rnd, job=ws.root.name)
        keep_session(ws, node)
        store.event("node_added", node=node.id, kind=node.info["kind"], parent=plan.get("parent"), round=rnd, agent_cost=res["cost"])
        store.reindex()
    finally:
        _close(ws)
    await evaluate(store, node)


async def mutate(store: Store, ref, secs, rnd: int):
    ref.update(ref_count=ref.info.get("ref_count", 0) + 1)
    store.event("mutation_start", round=rnd, reference=ref.id, secondaries=[s.id for s in secs])
    ws = new_workspace(store, workspace.mutation, ref, secs)
    plan = {"op": "mutation", "round": rnd, "parent": ref.id, "refs": [s.id for s in secs]}
    _plan(ws, **plan)
    await _finish(store, ws, plan, await run_agent_job(ws))


async def create(store: Store, examples, rnd: int):
    store.event("creation_start", round=rnd, examples=[e.id for e in examples])
    ws = new_workspace(store, workspace.creation, examples)
    plan = {"op": "creation", "round": rnd, "parent": None, "refs": [e.id for e in examples]}
    _plan(ws, **plan)
    await _finish(store, ws, plan, await run_agent_job(ws))


async def resume(store: Store, root):
    """Finish a queue job left open by an earlier loop process."""
    plan = json.loads((root / "loop.json").read_text())
    ws = SimpleNamespace(root=root, child=root / "rw" / "child", ro=root / "ro", kind="evolver" if plan["op"] == "mutation" else "creator")
    store.event("job_resumed", round=plan["round"], job=root.name)
    if not (root / "job.json").exists():  # created but never posted: rebuild the post from the workspace on disk
        ws = workspace.Workspace(
            root,
            root / "ro",
            root / "rw",
            ws.kind,
            json.loads((root / "samples.json").read_text()) if (root / "samples.json").exists() else [],
            json.loads((root / "rw" / "child" / "node.json").read_text()),
        )
    await _finish(store, ws, plan, await run_agent_job(ws))


def open_jobs(store: Store):
    d = store.dir / "agent_jobs"
    if not d.exists():
        return []
    return [p for p in sorted(d.iterdir()) if (p / "loop.json").exists() and not json.loads((p / "loop.json").read_text())["closed"]]


async def topup(store: Store, node, rnd: int):
    node.update(front_rounds=0, ref_count=0)
    store.event("topup_start", round=rnd, node=node.id)
    await evaluate(store, node)


class Scheduler:
    """Steady state: `workers` jobs always in flight, so agent thinking time overlaps other jobs' Qwen measurement.

    Each job starts from the pool as it is at that moment. Every SLOTS job starts is one "epoch": front counters
    (epochs a node has spent in layer 1) are bumped then, so top-up timing doesn't depend on the worker count.
    """

    def __init__(self, store: Store, rng: random.Random, start_job: int = 0):
        self.store, self.rng, self.job = store, rng, start_job
        self.busy: set[str] = set()  # nodes being topped up

    def _epoch(self, el, ls):
        front = {n.id for n in ls[0]} if ls else set()
        for n in el:
            n.update(front_rounds=n.info.get("front_rounds", 0) + 1 if n.id in front else 0)
        self.store.event("epoch", job=self.job, eligible=len(el), front=sorted(front), layers=[[n.id for n in layer] for layer in ls])

    def next_job(self):
        nodes = self.store.nodes()
        ukey = self.rng.choice(["U", "U_rev"])
        el = pool.eligible(nodes)
        if not el:
            return None
        ls = pool.layers(el, ukey)
        if self.job % SLOTS == 0:
            self._epoch(el, ls)
        self.job += 1
        j = self.job
        self.store.event("job_start", job=j, surprise=ukey)
        due = [n for n in pool.due_topups(nodes) if n.id not in self.busy]
        if due:
            self.busy.add(due[0].id)
            return self._topup(due[0], j)
        if self.rng.random() < P_CREATOR:
            return create(self.store, self.rng.sample(el, min(N_EXAMPLES, len(el))), j)
        ref, secs = pool.choose(nodes, ukey, self.rng)
        return mutate(self.store, ref, secs, j)

    async def _topup(self, node, j):
        try:
            await topup(self.store, node, j)
        finally:
            self.busy.discard(node.id)


def interrupted(store: Store):
    """Children whose measurement an earlier process didn't finish (evaluate drops the partial batch)."""
    return [n for n in store.nodes() if n.info.get("kind") in ("child", "creator") and n.info.get("stage") in ("submitted", "measuring")]


async def remeasure(store: Store, node):
    store.event("measure_resumed", node=node.id)
    await evaluate(store, node)


def max_jobs(store: Store, default: int) -> int:
    """data/<run>/loop.json {"max_jobs": N} overrides the cap while the loop runs."""
    p = store.dir / "loop.json"
    try:
        return int(json.loads(p.read_text())["max_jobs"]) if p.exists() else default
    except (ValueError, KeyError, TypeError):
        return default


async def run_loop(run: str, jobs: int, workers: int = 6, seed: int = 0, backend: str = "queue"):
    """Run until `jobs` jobs have started in this run (counting earlier processes), `workers` in flight.
    Queue jobs left open by an earlier process are resumed first and count toward `workers`."""
    global BACKEND
    BACKEND = backend
    store = Store(run)
    events = store.dir / "events.jsonl"
    lines = (await asyncio.to_thread(events.read_text)).split("\n") if events.exists() else []
    done = sum(1 for line in lines if '"job_start"' in line)
    sched = Scheduler(store, random.Random(f"{seed}-{done}"), done)
    pending: set[asyncio.Task] = {asyncio.create_task(resume(store, r)) for r in open_jobs(store)}
    pending |= {asyncio.create_task(remeasure(store, n)) for n in interrupted(store)}
    while True:
        cap = max_jobs(store, jobs)
        while sched.job < cap and len(pending) < workers:
            coro = sched.next_job()
            if coro is None:
                store.event("loop_stopped", reason="no eligible nodes")
                break
            pending.add(asyncio.create_task(coro))
        if not pending:
            break
        finished, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for t in finished:
            if t.exception():
                store.event("job_error", error=repr(t.exception())[:1000])
        store.reindex()
    store.event("loop_finished", jobs=sched.job)
