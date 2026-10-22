"""A subset of hiring_resume_attrs arms (the same 15 edits as hiring_resume_attrs_selective, plus base) on only the 11
borderline weak-batch bodies (see hiring_resume_names_borderline). Fresh samples. Run with --n 33 (3 per body).
Positive = DECISION: YES.
"""

from evalcue.envs.hiring_resume_attrs_selective import EDITS
from evalcue.envs.hiring_resume_data.core import build, calibrated, detect

ARMS = ["base", *EDITS]


def items():
    return calibrated()


__all__ = ["ARMS", "build", "detect", "items"]
