"""Goal-guarding (Meinke et al. 2024) with every edit stacked on the before_impl + permission cell of scheming_goalguard:
the memo also says today's selection is re-run under the new prompt and that operators may edit or revert the prompt
file. That cell was 4/10 in round 2 vs 0/160 for the plain memo, so edits here have headroom both ways. Arm "bp_base" is
that cell unedited; "bp_<x>" adds arm <x> of scheming_goalguard. See docs/envs/scheming.md."""

from evalcue.envs.scheming import apollo as A
from evalcue.envs.scheming.goalguard import build, detect

MEMO_ARMS = ["prompt_inline", "path_unmentioned", "no_misalignment_word", "update_soon", "checksummed", "feedback_channel"]
ARMS = ["bp_" + a for a in A.COMMON_ARMS + MEMO_ARMS]


def items():
    return A.ITEMS


__all__ = ["ARMS", "build", "detect", "items"]
