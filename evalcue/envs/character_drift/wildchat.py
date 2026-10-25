"""character_drift.wildchat: 40 real WildChat-1M roleplay chats with GPT-3.5 / GPT-4 (see the family docstring in __init__.py).

data/wildchat_chats.json (from build_wildchat.py) holds the first k <= 5 exchanges of each chat and the user's real next message.
Unlike the Sonnet scenarios the user asked for the role in turn 1, and the old model's replies are the prefill. Arms:
    drift      the old model's turns, no system prompt
    qwensys    the same under Qwen's default system prompt (QWEN_SYS)
x follow-ups {inworld (the user's real next message), offtopic, identity} = 6 arms, named PREFILL_FOLLOWUP.
"""

import json
from functools import cache
from pathlib import Path

from evalcue.envs.character_drift import BEHAVIOR, FOLLOWUPS, QWEN_SYS, detect, episode, followup

__all__ = ["ARMS", "BEHAVIOR", "build", "detect", "items"]

DATA = Path(__file__).parent / "data"
PREFILLS = ["drift", "qwensys"]
ARMS = [f"{p}_{f}" for p in PREFILLS for f in FOLLOWUPS]


@cache
def chats() -> dict[str, dict]:
    return {c["id"]: c for c in json.loads((DATA / "wildchat_chats.json").read_text())}


def items():
    return list(chats())


def build(arm, item):
    prefill, kind = arm.split("_")
    c = chats()[item]
    turns = [(t["user"], t["assistant"]) for t in c["turns"]]
    follow = followup(kind, items().index(item), c["in_world_followup"])
    meta = {"character": c.get("character"), "followup_kind": kind, "followup": follow, "prefill": prefill, "wildchat_model": c["model"]}
    meta["conversation_hash"] = c["conversation_hash"]
    return episode(QWEN_SYS if prefill == "qwensys" else None, turns, follow, meta)
