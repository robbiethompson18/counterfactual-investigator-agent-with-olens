"""character_drift.sonnet: 40 Sonnet 5.5-written drift transcripts (see the family docstring in __init__.py).

Each scenario (data/sonnet_scenarios.json, from gen_sonnet.py) has 5-6 turns of {user, drift, default}: drift turn 1 is a normal
helpful reply, later turns drift further, the last two are fully in character. Prefill arms:
    drift      drifted turns, no system prompt
    qwensys    drifted turns under Qwen's default system prompt (QWEN_SYS)
    sysrp      drifted turns under the scenario's own system prompt telling the model to be that character
    default    the same user turns with plain default-voice replies (control)
x follow-ups {inworld, offtopic, identity} = 12 arms, named PREFILL_FOLLOWUP.
"""

import json
from functools import cache
from pathlib import Path

from evalcue.envs.character_drift import BEHAVIOR, FOLLOWUPS, QWEN_SYS, detect, episode, followup

__all__ = ["ARMS", "BEHAVIOR", "build", "detect", "items"]

DATA = Path(__file__).parent / "data"
PREFILLS = ["drift", "qwensys", "sysrp", "default"]
ARMS = [f"{p}_{f}" for p in PREFILLS for f in FOLLOWUPS]


@cache
def scenarios() -> dict[str, dict]:
    return {s["id"]: s for s in json.loads((DATA / "sonnet_scenarios.json").read_text())}


def items():
    return list(scenarios())


def build(arm, item):
    prefill, kind = arm.split("_")
    s = scenarios()[item]
    key = "default" if prefill == "default" else "drift"
    turns = [(t["user"], t[key]) for t in s["turns"]]
    system = {"drift": None, "default": None, "qwensys": QWEN_SYS, "sysrp": s["system_prompt"]}[prefill]
    follow = followup(kind, items().index(item), s["in_world_followup"])
    meta = {"character": s["character"], "followup_kind": kind, "followup": follow, "prefill": prefill}
    return episode(system, turns, follow, meta)
