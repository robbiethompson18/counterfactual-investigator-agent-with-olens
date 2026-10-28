"""Run an eval: a folder cfbench/evals/<name>/ that picks its own cases for the cfbench predictor (cfbench/evals/CLAUDE.md).

    uv run python -m cfbench.eval NAME --prep [--versions a,b] [--placebo DIR]   # its inputs (cases, reads) as a runs.py run
    uv run python -m cfbench.eval NAME --slug SLUG [--versions a,b] [--draws 3] [--limit N] [--conditions ...]
    uv run python -m cfbench.score data/runs/cfbench/<date>_<slug>

The eval's eval.py gives ENVS (the environments/ envs it is about, for the env viewer), CONDITIONS, MODEL_NAME
(None, or what to call the target in place of its id), prep(out, versions, placebo), reads(inputs) and cases(inputs,
versions) -> (prompt env, case row fields, base runs) per case. eval.md beside it, if any, goes into every prompt as
"## About this evaluation", before the task. Everything else is cfbench as usual: core.build_prompt, the transcript and
O-lens tools, predict.predict, one predictions.jsonl row per draw in data/runs/cfbench/<date>_<slug>/, synced to HF.
"""

import argparse
import asyncio
import importlib
import json
import random
import re
from pathlib import Path

import openai

from cfbench import olens
from cfbench.core import HARNESS_VERSION, OLENS_CONDITIONS, TOOL_BUDGET, TOOL_CONDITIONS, build_prompt, pick_examples
from cfbench.predict import EFFORT, predict
from cfbench.tools import transcript_tools
from counterfactual_investigator_agent_with_olens import openrouter
from counterfactual_investigator_agent_with_olens.runs import ROOT, finish, new_run

EVALS = Path(__file__).parent / "evals"


def hide(prompt: str, model: str, name: str) -> str:
    return prompt.replace(f"the AI model `{model}`", name).replace(f"`{model}`", name)


def with_notes(prompt: str, notes: str) -> str:
    if not notes:
        return prompt
    return prompt.replace("\n\n## Your task\n", f"\n\n## About this evaluation\n\n{notes.strip()}\n\n## Your task\n", 1)


def latest_inputs(name: str) -> Path:
    runs = sorted((ROOT / "cfbench_evals").glob(f"*_{name.replace('_', '-')}-inputs"))
    assert runs, f"no inputs run for {name}: run `python -m cfbench.eval {name} --prep` first"
    return runs[-1]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="a folder in cfbench/evals/")
    ap.add_argument("--prep", action="store_true", help="build the eval's inputs as a runs.py run (cfbench_evals/<date>_<name>-inputs)")
    ap.add_argument("--placebo", default="", help="--prep: a cfbench reads dir from another env, for olens_placebo_agentic")
    ap.add_argument("--inputs", default="", help="the inputs run dir (default: the newest)")
    ap.add_argument("--versions", default="", help="comma list of the eval's versions (default: all it has)")
    ap.add_argument("--conditions", default="", help="comma list (default: the eval's CONDITIONS)")
    ap.add_argument("--predictors", default="google/gemini-3.8-flash")
    ap.add_argument("--effort", default=EFFORT)
    ap.add_argument("--max-calls", type=int, default=TOOL_BUDGET)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="first N cases only (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--slug", default="")
    ap.add_argument("--description", default="")
    args = ap.parse_args()
    ev = importlib.import_module(f"cfbench.evals.{args.name}.eval")
    versions = args.versions.split(",") if args.versions else list(ev.VERSIONS)
    if args.prep:
        run = new_run(
            "cfbench_evals",
            f"{args.name.replace('_', '-')}-inputs",
            description=f"inputs for cfbench eval {args.name} ({', '.join(versions)}): cases + O-lens reads (cfbench/evals/{args.name})",
            models=[],
            params={"eval": args.name, "versions": versions, "placebo": args.placebo},
        )
        ev.prep(run, versions, Path(args.placebo))
        finish(run)
        return
    assert args.slug, "--slug"
    inputs = Path(args.inputs) if args.inputs else latest_inputs(args.name)
    notes = (EVALS / args.name / "eval.md").read_text() if (EVALS / args.name / "eval.md").exists() else ""
    conditions = args.conditions.split(",") if args.conditions else list(ev.CONDITIONS)
    predictors = args.predictors.split(",")
    reads = olens.Reads(ev.reads(inputs)) if set(OLENS_CONDITIONS) & set(conditions) else None
    fve_note = olens.fve_note(reads) if reads else ""
    cases = list(ev.cases(inputs, versions))[: args.limit or None]
    run = new_run(
        "cfbench",
        args.slug,
        description=args.description or f"eval {args.name} ({', '.join(versions)}): {len(cases)} cases, {', '.join(conditions)}",
        models=[{"name": p, "served_by": "openrouter"} for p in predictors],
        params={
            "env": f"eval:{args.name}",
            "eval": args.name,
            "envs": ev.ENVS,
            "inputs": str(inputs.relative_to(ROOT)),
            "versions": versions,
            "conditions": ",".join(conditions),
            "predictors": args.predictors,
            "model_name": ev.MODEL_NAME,
            "models": sorted({c["model"] for _, c, _ in cases}),
            "task": "rate",
            "effort": args.effort,
            "max_calls": args.max_calls,
            "draws": args.draws,
            "seed": args.seed,
            "limit": args.limit,
            "harness_version": HARNESS_VERSION,
            "eval_md": notes,
        },
    )
    out, slots = run / "predictions.jsonl", asyncio.Semaphore(args.concurrency)

    async def one(env, case: dict, base: list, condition: str, predictor: str, draw: int):
        shown = pick_examples(base, random.Random(args.seed + draw))
        tools, read = {}, None
        if condition in TOOL_CONDITIONS:
            tools = transcript_tools(base, chars=condition in OLENS_CONDITIONS)
        if condition in OLENS_CONDITIONS:
            assert reads is not None
            lens_tools, read = olens.tools(base, reads, placebo=condition == "olens_placebo_agentic", seed=args.seed + draw)
            tools |= lens_tools
        prompt = build_prompt(env, case["model"], base, shown, case["base_arm"], case["cf_arm"], condition, read, fve_note, args.max_calls)
        prompt = with_notes(hide(prompt, case["model"], ev.MODEL_NAME) if ev.MODEL_NAME else prompt, notes)
        async with slots:
            for attempt in range(3):
                try:
                    result = await predict(predictor, prompt, tools, args.max_calls if tools else 0, args.effort)
                    break
                except (openai.APIStatusError, openrouter.NoChoices, json.JSONDecodeError) as e:
                    print(f"{case['env']} {case['model']} {condition} draw {draw}: attempt {attempt} failed: {e}")
            else:
                return
        if ev.MODEL_NAME:  # the predictor must never have seen the model id
            assert not re.search(re.escape(case["model"]), json.dumps(result["messages"])), case["model"]
        row = {**case, "condition": condition, "harness_version": HARNESS_VERSION, "eval": args.name, "predictor": predictor}
        row |= {"draw": draw, "shown": [r.source for r in shown], **result}
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"{case['env']} {case['model']} {condition} draw {draw}: p={result['p']} ${result['cost']:.3f}")

    jobs = [one(env, case, base, c, p, d) for env, case, base in cases for c in conditions for p in predictors for d in range(args.draws)]
    try:
        await asyncio.gather(*jobs)
    except BaseException:
        finish(run, status="abandoned")
        raise
    finish(run)


if __name__ == "__main__":
    asyncio.run(main())
