"""Run predictors on (env, target model, base arm -> counterfactual arm) cases.

    uv run python -m cfbench.run funding_email --pairs arm_a:arm_b,arm_b:arm_a --draws 5 --tag smoke_fe
    uv run python -m cfbench.score data/runs/cfbench/smoke_fe.jsonl

Each draw shows the predictor a fresh one-with/one-without pick of base-arm transcripts (condition `transcript`), or only the
base arm's starting context (`no_transcript`). Rows go to data/runs/cfbench/TAG.jsonl.
"""

import argparse
import asyncio
import importlib
import json
import random
from pathlib import Path

from cfbench.core import CONDITIONS, build_prompt, pick_examples
from cfbench.predict import REASONING, predict


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--pairs", required=True, help="comma list of BASE_ARM:CF_ARM")
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B", help="the model whose rollouts we predict")
    ap.add_argument("--predictors", default="anthropic/claude-sonnet-5.5,anthropic/claude-haiku-5.5")
    ap.add_argument("--condition", choices=CONDITIONS, default="transcript")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    env = importlib.import_module(f"cfbench.envs.{args.env}")
    out = Path(f"data/runs/cfbench/{args.tag}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    slots = asyncio.Semaphore(args.concurrency)

    async def one(case: dict, base: list, predictor: str, draw: int):
        shown = pick_examples(base, random.Random(args.seed + draw))  # same transcripts for every predictor on a given draw
        prompt = build_prompt(env, args.model, base, shown, case["base_arm"], case["cf_arm"], args.condition)
        async with slots:
            result = await predict(predictor, prompt)
        row = {**case, "predictor": predictor, "draw": draw, "shown": [r.source for r in shown], **result}
        with out.open("a") as f:  # noqa: ASYNC230 - one small append per prediction, written as it lands so a crash keeps them
            f.write(json.dumps(row) + "\n")
        print(f"{case['base_arm']}->{case['cf_arm']} {predictor} draw {draw}: p={result['p']} ${result['cost']:.3f}")

    jobs = []
    for pair in args.pairs.split(","):
        base_arm, cf_arm = pair.split(":")
        base, cf = env.rollouts(base_arm, args.model), env.rollouts(cf_arm, args.model)
        case = {
            "env": args.env,
            "model": args.model,
            "base_arm": base_arm,
            "cf_arm": cf_arm,
            "condition": args.condition,
            "reasoning": REASONING,
            "k_base": sum(r.label for r in base),
            "n_base": len(base),
            "k_cf": sum(r.label for r in cf),
            "n_cf": len(cf),
        }
        print(case)
        jobs += [one(case, base, p, d) for p in args.predictors.split(",") for d in range(args.draws)]
    await asyncio.gather(*jobs)


if __name__ == "__main__":
    asyncio.run(main())
