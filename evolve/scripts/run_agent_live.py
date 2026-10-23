"""Live agent sessions (Sonnet + DS via OpenRouter; costs money). Run explicitly:

    EVOLVE_LIVE=1 .venv/bin/python scripts/run_agent_live.py [evolver|creator|novelty ...]

Writes the toy store and transcripts under data/agent_tests/.
"""

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_agent_offline import make_toy_node
from evolve.node import Node, changed_chars, unified_diff

from evolve import agent, build, novelty, workspace

OUT = Path(__file__).resolve().parent.parent / "data" / "agent_tests"


class _Store:  # a minimal Store stand-in over OUT/store/nodes
    def __init__(self, d):
        self.d = d

    def nodes(self):
        return [Node(p) for p in sorted(self.d.iterdir()) if (p / "node.json").exists()]


def _keep(ws, name):
    dest = OUT / name / "child"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(ws.child, dest)
    return dest


async def evolver(toy):
    ws = workspace.mutation(toy, [toy, toy], base=OUT)
    try:
        res = await agent.run_workspace(ws, max_turns=40, max_cost=2.0, log=OUT / "evolver" / "session.json")
        child = _keep(ws, "evolver")
    finally:
        ws.cleanup()
    errs = build.check(child)
    summary = {
        "submitted": res["submitted"],
        "turns": res["turns"],
        "cost": round(res["cost"], 3),
        "recheck_errors": errs,
        "real_chars": changed_chars(child / "baseline", child / "real"),
        "null_chars": changed_chars(child / "baseline", child / "null"),
    }
    print(json.dumps(summary, indent=1))
    print(unified_diff(toy.dir / "baseline", child / "baseline")[:3000])
    print("REAL:", unified_diff(child / "baseline", child / "real")[:1500])
    print("NULL:", unified_diff(child / "baseline", child / "null")[:1500])
    return res


async def creator(toy):
    ws = workspace.creation([toy] * 6, base=OUT)
    try:
        res = await agent.run_workspace(ws, max_turns=40, max_cost=2.0, log=OUT / "creator" / "session.json")
        child = _keep(ws, "creator")
    finally:
        ws.cleanup()
    print(
        json.dumps(
            {"submitted": res["submitted"], "turns": res["turns"], "cost": round(res["cost"], 3), "recheck_errors": build.check(child)},
            indent=1,
        )
    )
    print(build.card(child))
    return res


async def novelty_check(store):
    out = {}
    for name in ("creator", "evolver"):
        child = OUT / name / "child"
        if child.exists():
            out[name] = await novelty.check_novelty(child, store)
            print(name, json.dumps(out[name]))
    (OUT / "novelty.json").write_text(json.dumps(out, indent=1) + "\n")


async def main(which):
    OUT.mkdir(parents=True, exist_ok=True)
    toy = make_toy_node(OUT / "store" / "nodes" / "toy1")
    if "evolver" in which:
        await evolver(toy)
    if "creator" in which:
        await creator(toy)
    if "novelty" in which:
        await novelty_check(_Store(OUT / "store" / "nodes"))


if __name__ == "__main__":
    if not os.environ.get("EVOLVE_LIVE"):
        sys.exit("set EVOLVE_LIVE=1 to run live sessions (they cost money)")
    asyncio.run(main(sys.argv[1:] or ["evolver", "creator", "novelty"]))
