"""Run predictors on one (env, target model, base arm -> counterfactual arm) case.

    uv run python -m cfbench.run funding_email --pair arm_a:arm_b --draws 5 --tag smoke
    uv run python -m cfbench.score data/runs/cfbench/smoke.jsonl

Each draw shows the predictor a fresh one-with/one-without pick of base-arm transcripts. Rows go to data/runs/cfbench/TAG.jsonl.
"""

import argparse
import asyncio
import importlib
import json
import random
from pathlib import Path

from cfbench.core import build_prompt, pick_examples
from cfbench.predict import predict


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--pair", required=True, help="BASE_ARM:CF_ARM")
    ap.add_argument("--model", default="Qwen/Qwen3.6-27B", help="the model whose rollouts we predict")
    ap.add_argument("--predictors", default="anthropic/claude-sonnet-5.5,anthropic/claude-haiku-5.5")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    env = importlib.import_module(f"cfbench.envs.{args.env}")
    base_arm, cf_arm = args.pair.split(":")
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

    out = Path(f"data/runs/cfbench/{args.tag}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    async def one(predictor: str, draw: int):
        shown = pick_examples(base, random.Random(args.seed + draw))  # same transcripts for every predictor on a given draw
        result = await predict(predictor, build_prompt(env, args.model, base, shown, base_arm, cf_arm))
        row = {**case, "predictor": predictor, "draw": draw, "shown": [r.source for r in shown], **result}
        with out.open("a") as f:  # noqa: ASYNC230 - one small append per prediction, written as it lands so a crash keeps them
            f.write(json.dumps(row) + "\n")
        print(f"{predictor} draw {draw}: p={result['p']} ${result['cost']:.3f} {result['rationale']}")

    await asyncio.gather(*(one(p, d) for p in args.predictors.split(",") for d in range(args.draws)))


if __name__ == "__main__":
    asyncio.run(main())
