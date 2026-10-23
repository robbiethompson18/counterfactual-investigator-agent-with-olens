"""evolve <command> <run> ...

seed <run> [seed ids]        copy seeds/* into data/<run>/nodes and measure them (30/arm, 16 picks)
evaluate <run> <node>        measure (or top up) one node
loop <run> --jobs N --workers W   run the evolution loop (see loop.py)
status <run>                 one line per node
"""

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

from .evaluate import evaluate
from .store import Store

SEEDS = Path(__file__).resolve().parent.parent / "seeds"


async def seed(run, ids, conc):
    st = Store(run)
    ids = ids or sorted(p.name for p in SEEDS.iterdir() if (p / "node.json").exists())
    sem = asyncio.Semaphore(conc)

    async def one(sid):
        node = st.node(sid)
        if not node.dir.exists():
            shutil.copytree(SEEDS / sid, node.dir)
            node.update(parent=None, refs=[], kind="seed", round=0, stage="submitted")
            st.event("node_added", node=sid, kind="seed")
        if node.scores():
            return
        async with sem:
            await evaluate(st, node)

    st.reindex()
    await asyncio.gather(*(one(s) for s in ids))


def status(run):
    for n in Store(run).nodes():
        i, r = n.info, n.scores().get("rank") or {}
        rates = n.scores().get("pooled_rates") or {}
        print(
            f"{n.id:48} {i.get('kind', ''):7} {i.get('stage', ''):10} audit={i.get('audit_score')} "
            + " ".join(f"{a}={v[0]}/{v[1]}" for a, v in rates.items())
            + (f"  L={r['L']:.2f} U={r['U']:.2f} U'={r['U_rev']:.2f} N={r['N']:.2f} P+={r['P_pos']:.2f}" if r else "")
        )


def main():
    if sys.argv[1:2] == ["dash"]:  # evolve dash <run> [--port P] [--host H ...]: the live dashboard (dashboard.py)
        from .dashboard import main as dash

        return dash(sys.argv[2:])
    ap = argparse.ArgumentParser(prog="evolve")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("seed")
    p.add_argument("run")
    p.add_argument("ids", nargs="*")
    p.add_argument("--conc", type=int, default=6)
    p = sub.add_parser("evaluate")
    p.add_argument("run")
    p.add_argument("node")
    p = sub.add_parser("loop")
    p.add_argument("run")
    p.add_argument("--jobs", type=int, default=40)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--backend", choices=["queue", "api"], default="queue", help="queue: external manager runs the agents")
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("status")
    p.add_argument("run")
    a = ap.parse_args()
    if a.cmd == "seed":
        asyncio.run(seed(a.run, a.ids, a.conc))
    elif a.cmd == "evaluate":
        st = Store(a.run)
        asyncio.run(evaluate(st, st.node(a.node)))
    elif a.cmd == "loop":
        from .loop import run_loop

        asyncio.run(run_loop(a.run, a.jobs, a.workers, a.seed, a.backend))
    else:
        status(a.run)


if __name__ == "__main__":
    main()
