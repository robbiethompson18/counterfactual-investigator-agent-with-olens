"""Lethal inaction (Lynch et al. 2025, "murder") with the single edits applied to the no-goal + replacement cell: no goal
in the system prompt, no goal conflict in the emails, only the replacement threat (mid-range for blackmail and leak).
Arm "base" is that cell unedited. The no-goal emails don't depend on the goal value, so the 8 items are identical here.
Prompts, detector and edits are the same as scheming/murder.py."""

from evalcue.envs.scheming import lynch as L
from evalcue.envs.scheming.murder import (
    BEHAVIOR,  # noqa: F401  (re-exported for cfbench)
    detect,
)

ARMS = L.ARMS_NG
items = L.items


def build(arm, item):
    return L.build("murder", arm, item, base=("none", "replacement"))


__all__ = ["ARMS", "build", "detect", "items"]
