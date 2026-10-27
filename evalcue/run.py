"""CLI: generate rows for an env, print every arm's rate, and sync the run to HF.

    uv run python -m evalcue.run ENV --tag dev [--n 10] [--arms a,b,c] [--thinking 1] [--seed 0] [-d "what this run tests"]
    uv run python -m evalcue.run ENV --rescore data/runs/ENV/DATE_TAG/rows.jsonl   # re-run detect() offline, then re-sync
Each generation run is its own scripts/runs.py run dir, data/runs/ENV/<YYYY-MM-DD>_TAG/ (rows.jsonl + meta.json), synced to
HF when it finishes, or as `abandoned` if it crashes. A tag can be used once per env per day.
"""

import argparse
import asyncio
import importlib
import json
import sys
from pathlib import Path

from counterfactual_investigator_agent_with_olens.runs import finish, new_run, upload
from evalcue import harness

ap = argparse.ArgumentParser()
ap.add_argument("env")
ap.add_argument("--n", type=int, default=10, help="samples per arm; start small, crank up for the best arms")
ap.add_argument("--arms", default=None, help="comma-separated; default env.ARMS")
ap.add_argument("--thinking", type=int, default=1)
ap.add_argument("--max-tokens", type=int, default=None, help="per generation; default: no cap beyond the 262k context. Don't cap thinking")
ap.add_argument("--tag", default="dev", help="run slug; unique per env per day")
ap.add_argument("-d", "--description", default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--concurrency", type=int, default=512)
ap.add_argument("--rescore", type=Path, default=None)
a = ap.parse_args()

env = importlib.import_module(f"evalcue.envs.{a.env}")
for line in Path("evalcue/envs/status.tsv").read_text().splitlines()[1:]:  # env status; unlisted envs are active
    name, status, reason, doc, date = line.split("\t")
    if name == a.env and status != "active":
        print(f"WARNING: {a.env} is {status} ({date}): {reason}. See {doc}", file=sys.stderr)
run_id = a.env.replace(".", "_")  # family envs (scheming.lynch) keep their flat pre-package run dir (scheming_lynch)
arms = a.arms.split(",") if a.arms else env.ARMS
if a.rescore:
    rows = harness.load_rows(a.rescore)
    for r in rows:
        for k in set(r) - harness.ROW_KEYS:  # old flat rows: drop detect()'s fields (and positive), then re-score
            del r[k]
        if "error" not in r:
            r.update(harness.score(env, r))
    a.rescore.write_text("".join(json.dumps(r) + "\n" for r in rows))
    if (a.rescore.parent / "meta.json").exists():
        upload(a.rescore.parent)
else:
    params = {"n_per_arm": a.n, "arms": arms, "thinking": bool(a.thinking), "max_tokens": a.max_tokens, "seed": a.seed}
    if harness.SAMPLING_MODE is not None:
        params["sampling"] = "thinking-on" if harness.SAMPLING_MODE else "thinking-off"
    served_by = f"vLLM at {harness.URL} (runpod:qwen-shared)"
    if harness.MODEL != "qwen":
        served_by = f"openrouter ({harness.PROVIDER or 'any provider'}, effort {harness.EFFORT})"
        if harness.RAW_PREFILL and not a.thinking:
            served_by = f"openrouter raw completions, empty think block prefilled (providers {','.join(harness.RAW_PROVIDERS)})"
    run = new_run(
        run_id,
        a.tag,
        description=a.description or f"evalcue {run_id}: {len(arms)} arms x n={a.n}",
        models=[{"name": "Qwen/Qwen3.6-27B" if harness.MODEL == "qwen" else harness.MODEL, "served_by": served_by}],
        params=params,
    )
    out = run / "rows.jsonl"
    try:
        asyncio.run(
            harness.run_batch(
                env,
                arms=arms,
                n_per_arm=a.n,
                thinking=bool(a.thinking),
                max_tokens=a.max_tokens,
                out=out,
                concurrency=a.concurrency,
                seed=a.seed,
            )
        )
    except BaseException:
        finish(run, status="abandoned")
        raise
    finish(run)
    rows = harness.load_rows(out)
    print(out)
harness.summarize([r for r in rows if r["arm"] in arms], arms)
