"""Copy one run's own nodes (not its seeds) into another run, e.g. a side run folded back into the main one.

    .venv/bin/python scripts/fold_run.py ns1 v1

Node dirs are copied as they are (ids are unique across runs); node.json gets "folded_from": "<src>". Parents must
already exist in the destination or be folded with them. Innocuousness ratings are per run, so the folded nodes have
none in the destination until scripts/backfill_innocuous.py gives them comparisons there.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evolve.store import Store


def main():
    src, dst = Store(sys.argv[1]), Store(sys.argv[2])
    have = {n.id for n in dst.nodes()}
    moved = []
    for n in src.nodes():
        if n.info.get("kind") == "seed" or n.id in have:
            continue
        if n.info.get("stage") != "scored" and not n.info.get("culled"):
            print(f"skip {n.id}: stage {n.info.get('stage')}")
            continue
        shutil.copytree(n.dir, dst.dir / "nodes" / n.id, symlinks=True)
        dst.node(n.id).update(folded_from=sys.argv[1])
        dst.event("node_added", node=n.id, kind=n.info.get("kind"), parent=n.info.get("parent"), folded_from=sys.argv[1])
        moved.append(n.id)
    missing = {dst.node(i).info.get("parent") for i in moved} - have - set(moved) - {None}
    dst.reindex()
    print(f"folded {len(moved)} nodes from {sys.argv[1]} into {sys.argv[2]}" + (f"; parents missing: {sorted(missing)}" if missing else ""))


if __name__ == "__main__":
    main()
