"""Run predictors on (env, target model, base arm -> counterfactual arm) cases.

    uv run python -m cfbench.run funding_email --pairs arm_a:arm_b,arm_b:arm_a --slug fe-smoke
    uv run python -m cfbench.score data/runs/cfbench/2026-10-21_fe-smoke

Each draw shows the predictor a fresh one-with/one-without pick of base-arm transcripts (condition `transcript`), or only the
base arm's starting context (`no_transcript`), or the starting context plus an index of every base-arm run and tools
to read or search them (`agentic`, cfbench/tools.py), plus O-lens tools over some of those runs' reads (`olens`)
or over another env's runs' reads (`olens_placebo`, cfbench/olens.py), or Thought-Anchors attention tools over the
runs' anchor reads (`anchors`, `anchors_placebo`, cfbench/anchors.py).
Every call is a scripts/runs.py run: predictions.jsonl plus meta.json in data/runs/cfbench/<date>_<slug>/, synced to HF
when it finishes (or fails).
"""

import argparse
import asyncio
import importlib
import json
import random
from pathlib import Path

import openai

from cfbench import anchors, olens
from cfbench.core import (
    ANCHOR_CONDITIONS,
    CONDITIONS,
    HARNESS_VERSION,
    OLENS_CONDITIONS,
    TOOL_BUDGET,
    TOOL_CONDITIONS,
    build_prompt,
    pick_examples,
)
from cfbench.predict import EFFORT, predict
from cfbench.tools import transcript_tools
from scripts.runs import finish, new_run


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--pairs", required=True, help="comma list of BASE_ARM:CF_ARM, or `hub` (envs with hub_pairs())")
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B", help="the model whose rollouts we predict")
    ap.add_argument("--predictors", default="anthropic/claude-haiku-5.5")
    ap.add_argument("--conditions", default="transcript,no_transcript", help=f"comma list of {CONDITIONS}")
    ap.add_argument("--olens-reads", default="", help="comma list of O-lens reads JSONL (cfbench/olens.py), for the olens conditions")
    ap.add_argument("--anchor-reads", default="", help="comma list of anchor reads JSONL (scripts/anchor_reads.py), for anchors*")
    ap.add_argument("--effort", default=EFFORT, help="predictor reasoning effort (OpenRouter); keep medium unless comparing")
    ap.add_argument("--max-calls", type=int, default=TOOL_BUDGET, help="tool conditions: read/search calls before it must submit")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--description", default="")
    ap.add_argument("--skip", default="", help="comma list of predictions.jsonl from earlier runs; skip the predictions they already have")
    args = ap.parse_args()
    key = lambda r: (r["base_arm"], r["cf_arm"], r["condition"], r["predictor"], r["draw"])
    done = {key(json.loads(line)) for path in filter(None, args.skip.split(",")) for line in Path(path).read_text().splitlines()}
    conditions, predictors = args.conditions.split(","), args.predictors.split(",")
    assert set(conditions) <= set(CONDITIONS), conditions
    reads = olens.Reads(args.olens_reads.split(",") if args.olens_reads else [])
    assert reads or not set(OLENS_CONDITIONS) & set(conditions), "the olens conditions need --olens-reads"
    fve_note = olens.fve_note(reads) if reads else ""
    anchor_reads = anchors.load(args.anchor_reads.split(",")) if args.anchor_reads else {}
    assert anchor_reads or not set(ANCHOR_CONDITIONS) & set(conditions), "the anchors conditions need --anchor-reads"

    module, _, spec = args.env.partition(":")  # `module:spec` for envs built from data, e.g. evalcue_runs:<env>/<run>
    env = importlib.import_module(f"cfbench.envs.{module}")
    env = env.load(spec) if spec else env
    pairs = env.hub_pairs() if args.pairs == "hub" else [tuple(pair.split(":")) for pair in args.pairs.split(",")]
    run = new_run(
        "cfbench",
        args.slug,
        description=args.description or f"{args.env}: predict {args.model} arm outcomes ({', '.join(conditions)})",
        models=[{"name": p, "served_by": "openrouter"} for p in predictors],
        params={k: v for k, v in vars(args).items() if k not in ("slug", "description")}
        | {"harness_version": HARNESS_VERSION}
        | ({"olens_sources": olens.sources(args.olens_reads.split(","))} if args.olens_reads else {})
        | ({"anchor_sources": anchors.sources(args.anchor_reads.split(","))} if args.anchor_reads else {}),
    )
    out = run / "predictions.jsonl"
    slots = asyncio.Semaphore(args.concurrency)

    async def one(case: dict, base: list, predictor: str, draw: int):
        shown = pick_examples(base, random.Random(args.seed + draw))  # same transcripts for every predictor on a given draw
        condition, tools, read = case["condition"], {}, None
        if condition in TOOL_CONDITIONS:
            tools = transcript_tools(base, chars=condition in OLENS_CONDITIONS + ANCHOR_CONDITIONS)
        if condition in OLENS_CONDITIONS:
            lens_tools, read = olens.tools(base, reads, placebo=condition == "olens_placebo", seed=args.seed + draw)
            tools |= lens_tools
        if condition in ANCHOR_CONDITIONS:
            anchor_tools, read = anchors.tools(base, anchor_reads, placebo=condition == "anchors_placebo", seed=args.seed + draw)
            tools |= anchor_tools
        prompt = build_prompt(env, args.model, base, shown, case["base_arm"], case["cf_arm"], condition, read, fve_note, args.max_calls)
        async with slots:
            for attempt in range(3):  # a rare provider 400 (Gemini: "reasoning details to be preserved") shouldn't kill the run
                try:
                    result = await predict(predictor, prompt, tools, args.max_calls if tools else 0, args.effort)
                    break
                except openai.APIStatusError as e:
                    print(f"{case['condition']} {case['base_arm']}->{case['cf_arm']} draw {draw}: attempt {attempt} failed: {e}")
            else:
                return  # left out of predictions.jsonl; rerun with --skip to fill it
        row = {
            **case,
            "harness_version": HARNESS_VERSION,
            "predictor": predictor,
            "draw": draw,
            "shown": [r.source for r in shown],
            **result,
        }
        with out.open("a") as f:  # appended as each lands, so a crash keeps them
            f.write(json.dumps(row) + "\n")
        print(f"{case['condition']} {case['base_arm']}->{case['cf_arm']} {predictor} draw {draw}: p={result['p']} ${result['cost']:.3f}")

    jobs = []
    for base_arm, cf_arm in pairs:
        base, cf = env.rollouts(base_arm, args.model), env.rollouts(cf_arm, args.model)
        case = {
            "env": args.env,
            "model": args.model,
            "base_arm": base_arm,
            "cf_arm": cf_arm,
            "k_base": sum(r.label for r in base),
            "n_base": len(base),
            "k_cf": sum(r.label for r in cf),
            "n_cf": len(cf),
        }
        print(case)
        jobs += [
            one(case | {"condition": c}, base, p, d)
            for c in conditions
            for p in predictors
            for d in range(args.draws)
            if (base_arm, cf_arm, c, p, d) not in done
        ]
    try:
        await asyncio.gather(*jobs)
    except BaseException:
        finish(run, status="abandoned")  # failed runs are data too
        raise
    finish(run)


if __name__ == "__main__":
    asyncio.run(main())
