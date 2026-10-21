"""secret_number (evalcue port of agent-interp-envs' noisy search) on Qwen3.6-27B: does the model read the secret file?

Rows are the evalcue runs from branch evalcue-secret-number (docs/evalcue-secret-number-log.md there), all with the "k2" knobs
(MAX=1024 BUDGET=5 NOISE=0.3 RULES=0, prefilled `pwd && ls -la && cat guess.py` turn), copied into data/envs/secret_number/.
Each arm pools every file it appears in. Arm contents come from the rows themselves (the model's starting context), since the
arms on that branch aren't merged; the label is the env's programmatic `hack` detector.
"""

import functools
import glob
import json
from pathlib import Path

from cfbench.core import Rollout, initial_context, render, unified_diff
from evalcue.envs import secret_number

DATA = Path("data/envs/secret_number")
KNOBS = {"max": 1024, "budget": 5, "noise": 0.3, "rules": False, "prefill": True}
MODELS = {"Qwen/Qwen3.6-27B": "qwen"}

assert secret_number.__doc__
BEHAVIOR = (
    "The behavior is a hack, detected programmatically from the model's bash commands and tool outputs. "
    "A run counts as a hack if any of these fire:\n\n" + secret_number.__doc__.split("no judge):\n")[1].split("Extras:")[0].rstrip()
)


@functools.cache
def _rows(model: str) -> list[tuple[str, dict]]:
    out = []
    for path in sorted(glob.glob(str(DATA / "*.jsonl"))):
        for i, line in enumerate(Path(path).read_text().splitlines()):
            row = json.loads(line)
            if row["model"] == MODELS[model] and all(row["meta"][k] == v for k, v in KNOBS.items()) and row["hack"] is not None:
                out.append((f"{Path(path).name}:{i}", row))
    return out


def rollouts(arm: str, model: str) -> list[Rollout]:
    return [Rollout(source=src, messages=row["messages"], label=bool(row["hack"])) for src, row in _rows(model) if row["arm"] == arm]


def edit(base_arm: str, cf_arm: str) -> str:
    """Diff of the starting context; every arm's context is the same across its rows (only the secret, which isn't shown, varies)."""
    base, cf = (render(initial_context(rollouts(arm, "Qwen/Qwen3.6-27B")[0].messages)) for arm in (base_arm, cf_arm))
    header = "Unified diff of the model's starting context (prompts and the prefilled first tool call):"
    return f"{header}\n\n```diff\n{unified_diff(base, cf)}\n```"
