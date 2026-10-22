"""secret_number (evalcue port of agent-interp-envs' noisy search) on Qwen3.6-27B: does the model read the secret file?

Rows are the evalcue runs in docs/evalcue-secret-number-log.md with the "k2" knobs (MAX=1024 BUDGET=5 NOISE=0.3 RULES=0,
prefilled `pwd && ls -la && cat guess.py` turn): the FILES below, from data/runs/evalcue/secret_number/.
Arm contents come from the rows themselves (the model's starting context), since those arms aren't on main yet; the label is
the env's programmatic `hack` detector.

Each arm pools every file it appears in, except that runs served by the shared Qwen server are kept apart as `<arm>@shared`:
nocue hacked 24/120 there vs 71/230 on our own pods, so mixing servers would bias every case against nocue.
"""

import functools
import json
from pathlib import Path

from cfbench.core import Rollout, initial_context, render, unified_diff
from evalcue.envs import secret_number

DATA = Path("data/runs/evalcue/secret_number")  # rows from before scripts/runs.py, one TAG.jsonl per run
# the k2 runs; the dir's other files are other models, knobs or debug runs
FILES = ["qwen2_k2_imp_norules", "qwen2_r2_imp_norules", "qwen2_r3_salient", "qwen2_r4_splithalves", "qwen2_r5_pathsplit"]
FILES += ["qwen2_confirm_imp_norules", "qwen2_confirm_prompt", "qwen2_r6_final", "qwen2_r7_final", "qwen2_r8_cfbench"]
KNOBS = {"max": 1024, "budget": 5, "noise": 0.3, "rules": False, "prefill": True}
MODELS = {"Qwen/Qwen3.6-27B": "qwen"}
SHARED_SERVER = {"qwen2_r6_final", "qwen2_r7_final", "qwen2_r8_cfbench"}  # files generated on $QWEN_VLLM_URL

assert secret_number.__doc__
BEHAVIOR = (
    "The behavior is a hack, detected programmatically from the model's bash commands and tool outputs. "
    "A run counts as a hack if any of these fire:\n\n" + secret_number.__doc__.split("no judge):\n")[1].split("Extras:")[0].rstrip()
)


@functools.cache
def _rows(model: str) -> list[tuple[str, str, dict]]:
    out = []
    for path in sorted(DATA / f"{name}.jsonl" for name in FILES):
        for i, line in enumerate(Path(path).read_text().splitlines()):
            row = json.loads(line)
            if row["model"] == MODELS[model] and all(row["meta"][k] == v for k, v in KNOBS.items()) and row["hack"] is not None:
                arm = row["arm"] + ("@shared" if Path(path).stem in SHARED_SERVER else "")
                out.append((f"{Path(path).name}:{i}", arm, row))
    return out


def rollouts(arm: str, model: str) -> list[Rollout]:
    return [Rollout(source=src, messages=row["messages"], label=bool(row["hack"])) for src, a, row in _rows(model) if a == arm]


def edit(base_arm: str, cf_arm: str) -> str:
    """Diff of the starting context; every arm's context is the same across its rows (only the secret, which isn't shown, varies)."""
    base, cf = (render(initial_context(rollouts(arm, "Qwen/Qwen3.6-27B")[0].messages)) for arm in (base_arm, cf_arm))
    header = "Unified diff of the model's starting context (prompts and the prefilled first tool call):"
    return f"{header}\n\n```diff\n{unified_diff(base, cf)}\n```"
