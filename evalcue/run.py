"""CLI: generate rows for an env and print the pre-registered stats.

    uv run python -m evalcue.run ENV [--n 50] [--arms nocue,cue,placebo] [--thinking 1] [--tag dev] [--seed 0]
    uv run python -m evalcue.run ENV --rescore data/runs/evalcue/ENV/dev.jsonl   # re-run detect() offline, no generation
Rows go to data/runs/evalcue/ENV/TAG.jsonl (appends).
"""

import argparse
import asyncio
import importlib
import json
from pathlib import Path

from evalcue import harness

ap = argparse.ArgumentParser()
ap.add_argument("env")
ap.add_argument("--n", type=int, default=50, help="samples per arm")
ap.add_argument("--arms", default=None, help="comma-separated; default env.ARMS (first = baseline)")
ap.add_argument("--thinking", type=int, default=1)
ap.add_argument("--max-tokens", type=int, default=None, help="per generation; default 8192 thinking / 2048 not")
ap.add_argument("--tag", default="dev")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--concurrency", type=int, default=64)
ap.add_argument("--rescore", type=Path, default=None)
a = ap.parse_args()

env = importlib.import_module(f"evalcue.envs.{a.env}")
arms = a.arms.split(",") if a.arms else env.ARMS
if a.rescore:
    rows = harness.load_rows(a.rescore)
    for r in rows:
        if "error" not in r:
            r.update(harness.score(env, r))
    a.rescore.write_text("".join(json.dumps(r) + "\n" for r in rows))
else:
    out = Path(f"data/runs/evalcue/{a.env}/{a.tag}.jsonl")
    max_tokens = a.max_tokens or (8192 if a.thinking else 2048)
    asyncio.run(
        harness.run_batch(
            env, arms=arms, n_per_arm=a.n, thinking=bool(a.thinking), max_tokens=max_tokens, out=out, concurrency=a.concurrency, seed=a.seed
        )
    )
    rows = [r for r in harness.load_rows(out) if r["seed"] == a.seed and r["thinking"] == bool(a.thinking)]
    print(out)
harness.summarize([r for r in rows if r["arm"] in arms], arms)
