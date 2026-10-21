"""funding_email from agent-interp-envs (2026): the agent polishes a Series B pitch; legal_review.md varies by arm.

Arms, labels and the screen report: docs/envs-screen/funding_email.md. Rollouts are pulled from robbie-box
`~/envs-results/funding_email/funding_email/<arm>/<model>/<ts>/run-N/` into data/envs/funding_email/ (last step's messages.json
and state.json, plus final/grading.json). Labels are Sonnet-subagent verdicts; runs without grading.json were invalid and are skipped.
"""

import glob
import json
from pathlib import Path

import yaml

from cfbench.core import Rollout
from scripts.envs_screen.funding_email_rubric import PROMPT, last_step

DATA = Path("data/envs/funding_email")

RUBRIC = PROMPT.split("## Transcript")[0].strip()
BEHAVIOR = f"A run shows the behavior when a grader following this rubric labels it `ethical_deviation`.\n\n<rubric>\n{RUBRIC}\n</rubric>"

SLOT = "the contents of the workspace file `legal_review.md`"
ARMS = {
    "arm_a": Path("vendor/agent-interp-envs/environments/funding_email/filesystem/legal_review.md").read_text(),  # shipped memo
    "arm_b": yaml.safe_load(Path("configs/envs_screen/funding_email/arm_b.yaml").read_text())["task"]["content"]["legal_review"],
}
MODELS = {"Qwen/Qwen3.6-27B": "Qwen-Qwen3.6-27B", "z-ai/glm-5.3": "z-ai-glm-5.3"}


def rollouts(arm: str, model: str) -> list[Rollout]:
    out = []
    for run in sorted(glob.glob(str(DATA / arm / MODELS[model] / "*" / "run-*"))):
        grading, step = Path(run) / "final/grading.json", last_step(Path(run))
        if not grading.exists() or step is None:
            continue
        label = json.loads(grading.read_text())["verdict"] == "ethical_deviation"
        out.append(Rollout(source=run, messages=json.loads((step / "messages.json").read_text()), label=label))
    return out
