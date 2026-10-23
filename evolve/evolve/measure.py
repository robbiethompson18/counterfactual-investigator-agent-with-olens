"""Run the target on a node's three arms and grade the runs.

A batch is n episodes per arm, saved as <node>/runs/<batch>/<arm>.jsonl (one record per line, with its verdict).
Batches are kept forever: the pooled rates use all of them, the Pool ranks on the latest batch (fresh draws).
"""

import asyncio
import json
import os
import random
import time

import openai

from .grading import grade_records, verdict
from .node import ARMS, Node
from .target import run_episode

TARGET_CONCURRENCY = asyncio.Semaphore(int(os.environ.get("EVOLVE_TARGET_CONC", "200")))


SERVER_DOWN = (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)
SERVER_WAIT = 2 * 3600  # how long an episode waits for the Qwen server to come back before giving up


async def _episode(arm, i):
    """One episode. If the Qwen server is unreachable the episode is restarted from scratch until it comes back
    (an outage must not turn into unscorable runs); any other failure is kept as an error marker."""
    t0 = time.time()
    while True:
        async with TARGET_CONCURRENCY:
            try:
                return await run_episode(arm)
            except SERVER_DOWN as e:
                if time.time() - t0 > SERVER_WAIT:
                    return {"error": f"server down for {SERVER_WAIT}s: {e!r}"[:500], "sample": i}
            except Exception as e:  # noqa: BLE001 (sandbox or other failure: keep a marker, never a fake verdict)
                return {"error": repr(e)[:500], "sample": i}
        await asyncio.sleep(30 + 30 * random.random())


async def run_batch(node: Node, n: int, batch: str | None = None) -> str:
    batch = batch or f"b{len(batches(node))}"
    out = node.dir / "runs" / batch
    out.mkdir(parents=True, exist_ok=True)
    arms = {a: node.arm(a) for a in ARMS}
    image = arms["baseline"].episode.get("image", "evalcue-sandbox")
    recs = dict(zip(ARMS, await asyncio.gather(*(asyncio.gather(*(_episode(arms[a], i) for i in range(n))) for a in ARMS))))
    for a in ARMS:
        gradable = lambda r: "error" not in r and r.get("finish") != "length"  # truncated runs are unscorable
        ok = [r for r in recs[a] if gradable(r)]
        vs = iter(await grade_records(node.grader_src, ok, image=image))
        rows = []
        for i, r in enumerate(recs[a]):
            v = next(vs) if gradable(r) else {"error": "episode failed" if "error" in r else "hit the token budget"}
            rows.append({"sample": i, "verdict": verdict(v), "grader_error": v.get("error") if isinstance(v, dict) else None, "record": r})
        await asyncio.to_thread(_write_jsonl, out / f"{a}.jsonl", rows)
    return batch


def _write_jsonl(path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)


def batches(node: Node) -> list[str]:
    d = node.dir / "runs"
    return sorted((p.name for p in d.iterdir() if p.is_dir()), key=lambda b: int(b[1:])) if d.exists() else []


def load(node: Node, batch: str, arm: str) -> list[dict]:
    p = node.dir / "runs" / batch / f"{arm}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def counts(node: Node, which=None) -> dict:
    """{arm: (k, n, unscorable)} over the given batches (default: all)."""
    out = {}
    for a in ARMS:
        rows = [r for b in (which or batches(node)) for r in load(node, b, a)]
        k = sum(r["verdict"] is True for r in rows)
        n = sum(r["verdict"] in (True, False) for r in rows)
        out[a] = (k, n, len(rows) - n)
    return out
