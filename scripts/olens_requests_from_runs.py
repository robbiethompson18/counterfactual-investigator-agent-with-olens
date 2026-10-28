"""Append O-lens read requests (scripts/olens_read_requests.py's input) for evalcue run rows.

    uv run python scripts/olens_requests_from_runs.py OUT.jsonl ENV/RUN[:arm,arm] ... [--per-arm 20] [--priority 1] [--layers 24,32]
    uv run python scripts/olens_requests_from_runs.py OUT.jsonl envs:ENV/RUN[:arm,arm] ... [--items 6031,6598] [--positions all]

ENV is the run dir's name (`scheming_murder_ng`), and RUN a dir under data/runs/ENV/; `envs:ENV/RUN` is an environments/
run folder under data/envs/ENV/ (cfbench/envs/env_runs.py). Rows with an error or no label are skipped, as cfbench skips
them; ids already in OUT are skipped, so rerunning adds only new rows. Ids are cfbench's `ENV/RUN:LINE` (0-based) or
`ENV/RUN:<episode id>`, so reads join to cfbench sources. --items keeps only those items; --positions all asks the reader
for every token instead of stop tokens (the dense-reads experiment, docs/envs/hint_cf.md). Tools are the ones the model
saw: the env's own schemas for tool-loop envs, none for single completions. --per-arm N takes a seeded random N of each
arm's rows per run (rows already in OUT count toward N), so a read budget covers every arm evenly.
"""

import argparse
import json
import random
from pathlib import Path

from cfbench.envs import env_runs
from environments._lib.core import load_env

# evalcue's bash tool schema (evalcue/harness.py, deleted 2026-10-27), which olens_read_requests.py knows by the name below
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command in the working directory. Returns stdout+stderr (truncated) and the exit code.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    },
}

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("specs", nargs="+")
ap.add_argument("--priority", type=int, default=1)
ap.add_argument("--per-arm", type=int, default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--layers", default=None, help="per-row layers (a subset of the reader's --layers); default: the reader's")
ap.add_argument("--items", default=None, help="comma list of items to keep")
ap.add_argument("--positions", default=None, choices=["all"], help="read every token, not just stop tokens")
a = ap.parse_args()

have = {json.loads(line)["id"] for line in a.out.open()} if a.out.exists() else set()
n = 0
with a.out.open("a") as f:
    for spec in a.specs:
        env_run = spec.startswith("envs:")
        run, _, arms = spec.removeprefix("envs:").partition(":")
        env_dir, run_dir = run.split("/")
        try:
            env = load_env(env_dir)  # the environments/ port, if this env has one (its build() matches the old rows)
        except ModuleNotFoundError:  # retired env (code deleted 2026-10-27): single-turn rows need no tools; others can't be read
            env = None
        if env_run:
            rows = [row for rs in env_runs.load(f"{env_dir}/{run_dir}").rows.values() for row in rs]  # (source, episode)
        else:
            lines = list(enumerate((Path("data/runs") / run / "rows.jsonl").read_text().splitlines()))
            rows = [(f"{run}:{i}", json.loads(line)) for i, line in lines]
        rows = [(rid, r) for rid, r in rows if "error" not in r and r.get("positive") is not None]
        rows = [(rid, r) for rid, r in rows if not a.items or str(r["item"]) in a.items.split(",")]
        rows = [(rid, r) for rid, r in rows if not arms or r["arm"] in arms.split(",")]
        if a.per_arm:
            random.Random(f"{a.seed}:{run}").shuffle(rows)
            taken: dict[str, int] = {}
            for rid, r in sorted(rows, key=lambda x: x[0] not in have):  # rows already requested come first
                taken[r["arm"]] = taken.get(r["arm"], 0) + 1
                r["keep"] = taken[r["arm"]] <= a.per_arm
            rows = [(rid, r) for rid, r in rows if r.pop("keep")]
        for rid, row in rows:
            if rid in have:
                continue
            if env_run:
                tools = row["tools"]  # recorded on the episode
            elif env is not None:
                ep = env.build(row["arm"], row["item"])
                tools = ep.tools or None  # single completions are sent without tools
            elif not row.get("files_before") and not any(m["role"] == "tool" for m in row["messages"]):
                tools = None
            else:
                raise SystemExit(f"{env_dir}: retired env with tool calls; its tool schemas went with its code (see environments/retired/)")
            req = {"id": rid, "env": env_dir, "arm": row["arm"], "item": row["item"], "positive": row["positive"]}
            req |= {"thinking": row.get("thinking"), "priority": a.priority, "messages": row["messages"]}
            req["tools"] = "evalcue.harness.BASH_TOOL" if tools == [BASH_TOOL] else tools
            if a.layers:
                req["layers"] = [int(x) for x in a.layers.split(",")]
            if a.positions:
                req["positions"] = a.positions
            f.write(json.dumps(req) + "\n")
            n += 1
print(f"appended {n} requests to {a.out}")
