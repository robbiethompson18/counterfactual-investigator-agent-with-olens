"""Oversight subversion (Meinke et al. 2024) with every edit stacked on the "blocking" cell of scheming_oversight: the memo
says non-developer-goal selections are blocked while oversight is enabled, and the selector really refuses them. That
cell was 5/10 in round 2 vs 2/160 for the plain memo (where nothing stops the model from just picking its own goal's
strategy), so edits here have headroom both ways. Arm "blk_base" is that cell unedited; "blk_<x>" adds arm <x> of
scheming_oversight. See docs/envs/scheming.md."""

from evalcue.envs.scheming import apollo as A
from evalcue.envs.scheming.oversight import build, detect

MEMO_ARMS = ["config_inline", "config_unmentioned", "config_audited", "threat_none", "threat_delete", "permission"]
ARMS = ["blk_" + a for a in A.COMMON_ARMS + MEMO_ARMS]


def items():
    return A.ITEMS


__all__ = ["ARMS", "build", "detect", "items"]
