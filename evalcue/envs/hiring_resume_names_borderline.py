"""hiring_resume_names on only the borderline weak-batch bodies: the 11 of 48 bodies whose decision was not unanimous across
the 16 name arms in run hiring_resume_names/2026-10-21_r2_weak_n48 (37 of 48 were all-YES or all-NO, so they carry no signal
about names). Fresh samples, so selecting on that run doesn't bias these rates. Run with --n = 3 x 11 = 33.
Positive = DECISION: YES.
"""

from evalcue.envs.hiring_resume_data.core import build, calibrated, detect
from evalcue.envs.hiring_resume_names import ARMS


def items():
    return calibrated()


__all__ = ["ARMS", "build", "detect", "items"]
