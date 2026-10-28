"""Run predictors on (env, target model, base arm -> counterfactual arm) cases.

    uv run python -m cfbench.run env_runs:<env>/<run> --pairs hub --conditions no_transcript,transcript_agentic --slug <slug>
    uv run python -m cfbench.score data/runs/cfbench/<date>_<slug>

Each draw shows the predictor only the base arm's starting context (`no_transcript`), or the starting context plus an index
of every base-arm run and tools to read or search them (`transcript_agentic`, cfbench/tools.py), plus O-lens tools over
some of those runs' reads (`olens_agentic`) or over another env's runs' reads (`olens_placebo_agentic`, cfbench/olens.py).
The retired inline two-transcript condition and the old names are in cfbench/core.py (LEGACY_CONDITIONS).
Every call is a scripts/runs.py run: predictions.jsonl plus meta.json in data/runs/cfbench/<date>_<slug>/, synced to HF
when it finishes (or fails).

`--task answer` asks for the model's answer in the counterfactual arm instead of P(behavior) (cfbench/answer.py; the env
needs `canon_answer`), scored by `cfbench.answer`, never `cfbench.score`. `--items all` (or a comma list) makes one case
per item, each env spec getting `@<item>`, for envs whose items are separate things to predict.
"""

import argparse
import asyncio
import importlib
import json
import random
from collections import Counter
from pathlib import Path

import openai

from cfbench import olens
from cfbench.core import (
    CONDITIONS,
    HARNESS_VERSION,
    OLENS_CONDITIONS,
    TASKS,
    TOOL_BUDGET,
    TOOL_CONDITIONS,
    build_prompt,
    pick_examples,
)
from cfbench.predict import EFFORT, predict
from cfbench.tools import transcript_tools
from counterfactual_investigator_agent_with_olens import openrouter
from counterfactual_investigator_agent_with_olens.runs import finish, new_run

# The olens conditions refuse a base arm where fewer than this many runs and under this share of its runs have reads
# (2026-10-27: 23 of 268 olens runs had such arms, down to 1 read of 64, and nothing flagged it). Agents pass
# --allow-few-reads only after a human approves.
MIN_READS, MIN_READ_FRAC = 5, 0.5


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--pairs", required=True, help="comma list of BASE_ARM:CF_ARM, or `hub` (envs with hub_pairs())")
    ap.add_argument("--task", default="rate", choices=TASKS, help="rate: P(behavior) in the cf arm; answer: the model's answer there")
    ap.add_argument("--items", default="", help="one case per item: a comma list of items, or `all` (env_runs envs)")
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B", help="the model whose rollouts we predict")
    ap.add_argument("--predictors", default="google/gemini-3.8-flash", help="comma list of OpenRouter ids")
    ap.add_argument("--conditions", default="no_transcript,transcript_agentic", help=f"comma list of {CONDITIONS}")
    ap.add_argument("--olens-reads", default="", help="comma list of O-lens reads JSONL (cfbench/olens.py), for the olens conditions")
    ap.add_argument("--allow-few-reads", action="store_true", help="run olens conditions on arms with few reads; agents ask a human first")
    ap.add_argument("--olens-before", help="regex; hide every O-lens read at or after its first match in each run's transcript (harness 8)")
    ap.add_argument(
        "--olens-positions",
        default="sparse",
        choices=olens.POSITIONS,
        help="sparse: every read of a non-dense request, and only template and stop tokens of an every-token one; all: every "
        'token, and every transcript used must have been read at every token (positions: "all" in its requests.jsonl)',
    )
    ap.add_argument("--effort", default=EFFORT, help="predictor reasoning effort (OpenRouter) or off; keep medium unless comparing")
    ap.add_argument("--max-calls", type=int, default=TOOL_BUDGET, help="tool conditions: read/search calls before it must submit")
    ap.add_argument(
        "--base-positive", action="store_true", help="show only base-arm runs that showed the behavior (answer task: hint followers)"
    )
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--description", default="")
    ap.add_argument("--skip", default="", help="comma list of predictions.jsonl from earlier runs; skip the predictions they already have")
    args = ap.parse_args()
    done = {  # (env, base_arm, cf_arm, condition, predictor, draw) already predicted by the --skip runs
        tuple(r[k] for k in ("env", "base_arm", "cf_arm", "condition", "predictor", "draw"))
        for path in filter(None, args.skip.split(","))
        for r in map(json.loads, Path(path).read_text().splitlines())
    }
    conditions, predictors = args.conditions.split(","), args.predictors.split(",")
    # the only version-8 change is --olens-before; without it the tools are version 7, so rows stay comparable with 7
    harness_version = HARNESS_VERSION if args.olens_before else 7
    assert set(conditions) <= set(CONDITIONS), conditions
    reads = olens.Reads(args.olens_reads.split(",") if args.olens_reads else [], args.olens_positions)
    assert reads or not set(OLENS_CONDITIONS) & set(conditions), "the olens conditions need --olens-reads"
    fve_note = olens.fve_note(reads) if reads else ""

    module, _, spec = args.env.partition(":")  # `module:spec` for envs built from data, e.g. env_runs:<env>/<run>
    loader = importlib.import_module(f"cfbench.envs.{module}")
    env = loader.load(spec) if spec else loader
    pairs = env.hub_pairs() if args.pairs == "hub" else [tuple(pair.split(":")) for pair in args.pairs.split(",")]
    if args.items:
        items = env.items() if args.items == "all" else args.items.split(",")
        envs = [(f"{module}:{spec}@{item}", loader.load(f"{spec}@{item}")) for item in items]
    else:
        envs = [(args.env, env)]
    if args.task == "answer":
        assert all(hasattr(e, "canon_answer") for _, e in envs), f"{args.env} has no canon_answer; the answer task needs one"
    if set(OLENS_CONDITIONS) & set(conditions) and args.olens_positions == "all":
        sparse_only = [
            f"{r.source}  ({reads.sparse_only[r.source]})"
            for env_name, e in envs
            for base_arm in dict.fromkeys(b for b, _ in pairs)
            for r in e.rollouts(base_arm, args.model)
            if r.source in reads.sparse_only
        ]
        assert not sparse_only, (
            '--olens-positions all, but these base runs were not read at every token (no positions: "all" in requests.jsonl):\n'
            + "\n".join(sparse_only)
        )
    if set(OLENS_CONDITIONS) & set(conditions) and not args.allow_few_reads:
        few = [
            f"{env_name} {base_arm}: {k} of {len(base)}"
            for env_name, e in envs
            for base_arm in dict.fromkeys(b for b, _ in pairs)
            for base in [e.rollouts(base_arm, args.model)]
            if (k := sum(r.source in reads for r in base)) < MIN_READS and k < MIN_READ_FRAC * len(base)
        ]
        assert not few, (
            f"base arms where fewer than {MIN_READS} runs and under {MIN_READ_FRAC:.0%} have O-lens reads. Read more runs; agents: "
            "ask a human before passing --allow-few-reads:\n" + "\n".join(few)
        )
    run = new_run(
        "cfbench",
        args.slug,
        description=args.description or f"{args.env}: predict {args.model} arm outcomes ({', '.join(conditions)})",
        models=[{"name": p, "served_by": "openrouter"} for p in predictors],
        params={k: v for k, v in vars(args).items() if k not in ("slug", "description")}
        | {"harness_version": harness_version}
        | getattr(env, "meta", {})  # e.g. env_runs: which label file each run used
        | ({"olens_sources": olens.sources(args.olens_reads.split(","))} if args.olens_reads else {}),
    )
    out = run / "predictions.jsonl"
    slots = asyncio.Semaphore(args.concurrency)

    async def one(env, case: dict, base: list, predictor: str, draw: int):
        shown = pick_examples(base, random.Random(args.seed + draw), args.task)  # same transcripts for every predictor on a given draw
        condition, tools, read = case["condition"], {}, None
        if condition in TOOL_CONDITIONS:
            tools = transcript_tools(base, chars=condition in OLENS_CONDITIONS)
        if condition in OLENS_CONDITIONS:
            lens_tools, read = olens.tools(
                base, reads, placebo=condition == "olens_placebo_agentic", seed=args.seed + draw, before=args.olens_before
            )
            tools |= lens_tools
        prompt = build_prompt(
            env, args.model, base, shown, case["base_arm"], case["cf_arm"], condition, read, fve_note, args.max_calls, args.task
        )
        async with slots:
            for attempt in range(3):  # a rare provider 400 (Gemini: "reasoning details to be preserved") shouldn't kill the run
                try:
                    result = await predict(predictor, prompt, tools, args.max_calls if tools else 0, args.effort, args.task)
                    break
                except (
                    openai.APIStatusError,
                    openai.APIConnectionError,
                    openrouter.NoChoices,
                    json.JSONDecodeError,
                ) as e:  # timeouts reach here unretried
                    print(f"{case['condition']} {case['base_arm']}->{case['cf_arm']} draw {draw}: attempt {attempt} failed: {e}")
            else:
                return  # left out of predictions.jsonl; rerun with --skip to fill it
        if args.task == "answer":
            canon = env.canon_answer(result["answer"]) if result["valid"] else None
            result |= {"canon": canon, "correct": canon == case["modal_cf"] if result["valid"] else None}
        row = {
            **case,
            "harness_version": harness_version,
            "predictor": predictor,
            "draw": draw,
            "shown": [r.source for r in shown],
            **result,
        }
        with out.open("a") as f:  # appended as each lands, so a crash keeps them
            f.write(json.dumps(row) + "\n")
        got = (
            f"answer={result['answer']!r} ({result['canon']}, {'correct' if result['correct'] else 'wrong'})"
            if args.task == "answer"
            else f"p={result['p']}"
        )
        print(f"{case['condition']} {case['base_arm']}->{case['cf_arm']} {predictor} draw {draw}: {got} ${result['cost']:.3f}")

    jobs = []
    for env_name, env in envs:
        for base_arm, cf_arm in pairs:
            base, cf = env.rollouts(base_arm, args.model), env.rollouts(cf_arm, args.model)
            if args.base_positive:  # e.g. hint_mcq's answer task: a base run that ignored the hint shows the own answer in the index
                base = [r for r in base if r.label]
            assert base and cf, f"{env_name} {base_arm}->{cf_arm}: no rollouts of {args.model} ({len(base)}, {len(cf)}); check --model"
            case = {"env": env_name, "model": args.model, "task": args.task, "base_arm": base_arm, "cf_arm": cf_arm}
            if args.task == "answer":
                answers_base, answers_cf = Counter(r.answer for r in base), Counter(r.answer for r in cf)
                modal_cf, k_cf = answers_cf.most_common(1)[0]
                case |= {
                    "answers_base": dict(answers_base),
                    "answers_cf": dict(answers_cf),
                    "modal_cf": modal_cf,
                    "k_cf": k_cf,
                    "n_cf": len(cf),
                }
            else:
                case |= {"k_base": sum(r.label for r in base), "n_base": len(base), "k_cf": sum(r.label for r in cf), "n_cf": len(cf)}
            print(case)
            jobs += [
                one(env, case | {"condition": c}, base, p, d)
                for c in conditions
                for p in predictors
                for d in range(args.draws)
                if (env_name, base_arm, cf_arm, c, p, d) not in done
            ]
    try:
        await asyncio.gather(*jobs)
    except BaseException:
        finish(run, status="abandoned")  # failed runs are data too
        raise
    finish(run)


if __name__ == "__main__":
    asyncio.run(main())
