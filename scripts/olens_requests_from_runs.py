"""Append O-lens read requests (scripts/olens_read_requests.py's input) for evalcue run rows.

    uv run python scripts/olens_requests_from_runs.py OUT.jsonl ENV/RUN[:arm,arm] ... [--priority 1] [--layers 24,32]

ENV is the run dir's name (`scheming_murder_ng`), and RUN a dir under data/runs/ENV/. Rows with an error or no label are
skipped, as cfbench skips them; ids already in OUT are skipped, so rerunning adds only new rows. Ids are cfbench's
`ENV/RUN:LINE` (0-based), so reads join to cfbench sources. Tools are the ones the model saw: the env's own schemas for
tool-loop envs, none for single completions.
"""

import argparse
import json
from pathlib import Path

from evalcue import envs
from evalcue.harness import BASH_TOOL

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("specs", nargs="+")
ap.add_argument("--priority", type=int, default=1)
ap.add_argument("--layers", default=None, help="per-row layers (a subset of the reader's --layers); default: the reader's")
a = ap.parse_args()

have = {json.loads(line)["id"] for line in a.out.open()} if a.out.exists() else set()
n = 0
with a.out.open("a") as f:
    for spec in a.specs:
        run, _, arms = spec.partition(":")
        env_dir, run_dir = run.split("/")
        env = envs.load(env_dir)
        for i, line in enumerate((Path("data/runs") / run / "rows.jsonl").read_text().splitlines()):
            row = json.loads(line)
            rid = f"{run}:{i}"
            if rid in have or "error" in row or row.get("positive") is None or (arms and row["arm"] not in arms.split(",")):
                continue
            ep = env.build(row["arm"], row["item"])
            tools = ep.tools if ep.files or ep.step else None  # single completions are sent without tools
            req = {"id": rid, "env": env_dir, "arm": row["arm"], "item": row["item"], "positive": row["positive"]}
            req |= {"thinking": row.get("thinking"), "priority": a.priority, "messages": row["messages"]}
            req["tools"] = "evalcue.harness.BASH_TOOL" if tools == [BASH_TOOL] else tools
            if a.layers:
                req["layers"] = [int(x) for x in a.layers.split(",")]
            f.write(json.dumps(req) + "\n")
            n += 1
print(f"appended {n} requests to {a.out}")
