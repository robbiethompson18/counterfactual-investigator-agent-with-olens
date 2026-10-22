"""Calibration pass for the hiring_resume family: every drafted resume body (8 jobs x 18 strengths/weaknesses) under the
default context (company culture text), the plain posting, and the selectivity text. Used to pick borderline bodies
(see hiring_resume/data/calibrate.py); not a counterfactual env itself, though its arms are valid context arms.
Run with --n = 3 x len(items) to get 3 samples per body per arm.
"""

from evalcue.envs.hiring_resume.attrs import BEHAVIOR  # noqa: F401  (re-exported for cfbench)
from evalcue.envs.hiring_resume.data.core import bodies, build, detect

ARMS = ["base", "ctx_plain", "selective", "bar_half", "bar_quarter"]


def items():
    return sorted(b for b in bodies() if "_w" not in b)  # first batch only (the weak batch has hiring_resume_calib_weak)


__all__ = ["ARMS", "build", "detect", "items"]
