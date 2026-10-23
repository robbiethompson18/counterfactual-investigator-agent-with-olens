"""Give every scored node of a run at least K innocuousness comparisons (innocuous.py).

    .venv/bin/python scripts/backfill_innocuous.py v1 [--k 4] [--conc 6] [--max-cost 5] [--dry-run]

Pairs come in rounds of random cycles over the nodes that still need comparisons: one cycle connects them all
(plus one already-rated node, if any, to tie them to the existing graph) and gives each node one turn as A and one
as B. Later cycles avoid pairs already compared. Stops starting new comparisons once spend passes --max-cost.
"""

import argparse
import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evolve.store import Store

from evolve import innocuous


def plan(ids: list[str], counts: dict[str, int], done: set[frozenset], k: int, rng: random.Random) -> list[tuple[str, str]]:
    need = {i: max(0, k - counts.get(i, 0)) for i in ids}
    rated = [i for i in ids if counts.get(i, 0) > 0]
    pairs, used = [], set(done)
    first = True
    while any(need.values()):
        cyc = [i for i in ids if need[i] > 0]
        if first and rated and len(cyc) < len(ids):
            cyc.append(rng.choice([i for i in rated if i not in cyc] or rated))
        first = False
        if len(cyc) == 1:
            cyc.append(rng.choice([i for i in ids if i != cyc[0]]))
        best = None
        for _ in range(200):  # fewest repeated pairs
            rng.shuffle(cyc)
            edges = list(zip(cyc, cyc[1:] + cyc[:1])) if len(cyc) > 2 else [tuple(cyc)]
            rep = sum(frozenset(e) in used for e in edges)
            if best is None or rep < best[0]:
                best = (rep, edges)
            if rep == 0:
                break
        for a, b in best[1]:
            if not (need[a] or need[b]):
                continue
            pairs.append((a, b))
            used.add(frozenset((a, b)))
            for x in (a, b):
                need[x] = max(0, need[x] - 1)
    return pairs


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--k", type=int, default=innocuous.K)
    ap.add_argument("--conc", type=int, default=6)
    ap.add_argument("--max-cost", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    st = Store(a.run)
    nodes = {n.id: n for n in st.nodes() if n.scores().get("rank") and n.info.get("audit_score") is not None}
    rows = [r for r in innocuous.comparisons(st.dir) if r.get("winner")]
    counts = {i: sum(i in (r["a"], r["b"]) for r in rows) for i in nodes}
    pairs = plan(sorted(nodes), counts, {frozenset((r["a"], r["b"])) for r in rows}, a.k, random.Random(a.seed))
    print(f"{len(nodes)} scored nodes, {len(rows)} comparisons so far, {len(pairs)} to run")
    if a.dry_run:
        for p in pairs:
            print(*p)
        return
    sem, spent, n_done = asyncio.Semaphore(a.conc), [0.0], [0]

    async def one(x, y):
        async with sem:
            if spent[0] >= a.max_cost:
                return
            try:
                r = await innocuous.compare(st.dir, nodes[x], nodes[y])
            except Exception as e:  # noqa: BLE001
                print(f"FAILED {x} vs {y}: {e!r}", flush=True)
                return
            spent[0] += r["cost"]
            n_done[0] += 1
            print(f"[{n_done[0]}/{len(pairs)} ${spent[0]:.3f}] {x} vs {y} -> {r['winner']}", flush=True)

    await asyncio.gather(*(one(x, y) for x, y in pairs))
    st.event("innocuous_backfill", comparisons=n_done[0], cost=round(spent[0], 4))
    print(f"done: {n_done[0]} comparisons, ${spent[0]:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
