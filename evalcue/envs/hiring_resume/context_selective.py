"""hiring_resume_context's context arms in Karvonen & Marks's regime: first (strong) batch with the top-10% selectivity text
in every arm, x 5 contexts (plain posting, culture text without its diversity paragraph, anti-bias statement, eval cue,
deploy cue) x 4 names (White/Black x female/male). The bare-culture-text cells are hiring_resume_names_selective's.
Positive = DECISION: YES.
"""

from evalcue.envs.hiring_resume.context import CONTEXTS, NAMES
from evalcue.envs.hiring_resume.data.core import build, detect, interleaved

ARMS = [f"{n}+selective+{c}" for c in CONTEXTS for n in NAMES]


def items():
    return interleaved(weak=False)


__all__ = ["ARMS", "build", "detect", "items"]
