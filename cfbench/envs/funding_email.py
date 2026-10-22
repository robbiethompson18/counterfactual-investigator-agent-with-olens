"""funding_email from agent-interp-envs (2026): the agent polishes a Series B pitch; legal_review.md varies by arm.

Arms, labels and the screen report: docs/envs-screen/funding_email.md. Rollouts are the screen's run dir (synced from robbie-box
`~/envs-results/funding_email/`), `<arm>/<model>/<ts>/run-N/` under DATA; each run's final/grading.json holds its label.
Labels are Sonnet-subagent verdicts; runs without grading.json were invalid and are skipped.
"""

import glob
import json
from pathlib import Path

import yaml

from cfbench.core import Rollout
from cfbench.envs.agent_interp import rollout
from scripts.envs_screen.funding_email_rubric import PROMPT

DATA = Path("data/runs/funding_email/2026-10-21_screen/funding_email")

RUBRIC = PROMPT.split("## Transcript")[0].strip()
BEHAVIOR = f"A run shows the behavior when a grader following this rubric labels it `ethical_deviation`.\n\n<rubric>\n{RUBRIC}\n</rubric>"

SLOT = "the contents of the workspace file `legal_review.md`"
ARMS = {
    "arm_a": Path("vendor/agent-interp-envs/environments/funding_email/filesystem/legal_review.md").read_text(),  # shipped memo
    "arm_b": yaml.safe_load(Path("configs/envs_screen/funding_email/arm_b.yaml").read_text())["task"]["content"]["legal_review"],
}
MODELS = {"Qwen/Qwen3.6-27B": "Qwen-Qwen3.6-27B", "z-ai/glm-5.3": "z-ai-glm-5.3"}


def edit(base_arm: str, cf_arm: str) -> str:
    return f"""\
The only difference is {SLOT}.

Base arm version:
<base>
{ARMS[base_arm]}
</base>

Counterfactual arm version:
<counterfactual>
{ARMS[cf_arm]}
</counterfactual>"""


def rollouts(arm: str, model: str) -> list[Rollout]:
    out = []
    for run in sorted(Path(p) for p in glob.glob(str(DATA / arm / MODELS[model] / "*" / "run-*"))):
        grading = run / "final/grading.json"
        r = grading.exists() and rollout(run, json.loads(grading.read_text())["verdict"] == "ethical_deviation")
        if r:
            out.append(r)
    return out
