"""Hiring decisions from one borderline resume, name swapped: 7 race/ethnicity-coded groups x 2 genders (several names per
group, assigned across items), plus a redacted name. Default context (company culture text). Positive = DECISION: YES.
See docs/envs/hiring_resume.md and hiring_resume_data/core.py.
"""

from evalcue.envs.hiring_resume_data.core import GROUPS, build, calibrated, detect

ARMS = ["base", "name_none"] + [f"name_{g}_{s}" for g in GROUPS for s in "fm"]


def items():
    return calibrated()


__all__ = ["ARMS", "build", "detect", "items"]
