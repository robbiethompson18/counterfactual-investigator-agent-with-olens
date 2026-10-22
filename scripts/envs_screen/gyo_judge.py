"""Aditya's grade_your_own autorater, unchanged except for one missing import.

At the vendored pin (310c03f), environments/grade_your_own/grade_rollouts.py calls `yaml.safe_load` in
`run_target` without importing yaml, so every run crashes with NameError. This loads that module, injects
`yaml`, and runs its own judge_cli. Run from the harness dir (vendor/agent-interp-envs) with its uv env:

    uv run python <this repo>/scripts/envs_screen/gyo_judge.py <results dir> --model claude-sonnet-5-5
"""

import importlib.util
import sys

import yaml

spec = importlib.util.spec_from_file_location("grade_rollouts", "environments/grade_your_own/grade_rollouts.py")
assert spec is not None and spec.loader is not None
grader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grader)
grader.yaml = yaml  # ty: ignore[unresolved-attribute]

sys.exit(grader.judge_cli(grader.item_fn, rating_model=grader.RouteRating, description=grader.__doc__))
