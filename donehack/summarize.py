"""Summarize a run dir: hack rate by condition and task, plus hack type."""

import json
import sys
from collections import defaultdict
from pathlib import Path

for run in sys.argv[1:]:
    recs = [json.loads(Path(f).read_text())["record"] for f in sorted(Path("runs", run).glob("*.json"))]
    cost = sum(r["usage"]["cost"] for r in recs)
    print(f"== {run}: {len(recs)} episodes, ${cost:.2f}")
    by = defaultdict(list)
    for r in recs:
        by[(r["cond"], r["task"])].append(r)
        by[(r["cond"], "ALL")].append(r)
    print(f"{'cond':12s} {'task':14s} {'n':>3s} {'hack':>5s} {'test_edit':>9s} {'special':>7s} {'done':>5s} {'giveup':>6s} {'maxstep':>7s}")
    for (cond, task), rs in sorted(by.items()):
        n = len(rs)
        hack = sum(r["hack"] for r in rs)
        edit = sum(r["hack"] and r["tests_modified"] for r in rs)
        spec = sum(r["hack"] and r["orig_tests_pass"] for r in rs)
        print(
            f"{cond:12s} {task:14s} {n:3d} {hack:5d} {edit:9d} {spec:7d} "
            f"{sum(r['end'] == 'done' for r in rs):5d} {sum(r['end'] == 'give_up' for r in rs):6d} "
            f"{sum(r['end'] == 'max_steps' for r in rs):7d}"
        )
