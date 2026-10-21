"""Shared by the agent-interp-envs envs: a run dir is <...>/run-N/step-K/{messages,state}.json; the last step holds the transcript."""

import json
from pathlib import Path

from cfbench.core import Rollout
from scripts.envs_screen.funding_email_rubric import last_step  # harness-generic despite the module it lives in


def rollout(run: Path, label: bool) -> Rollout | None:
    step = last_step(run)
    return None if step is None else Rollout(source=str(run), messages=json.loads((step / "messages.json").read_text()), label=label)
