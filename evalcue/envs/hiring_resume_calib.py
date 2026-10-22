"""Calibration pass for the hiring_resume family: every drafted resume body (8 jobs x 18 strengths/weaknesses) under the
default context (company culture text), the plain posting, and the selectivity text. Used to pick borderline bodies
(see hiring_resume_data/calibrate.py); not a counterfactual env itself, though its arms are valid context arms.
Run with --n = 3 x len(items) to get 3 samples per body per arm.
"""

from evalcue.envs.hiring_resume_data.core import bodies, build, detect

ARMS = ["base", "ctx_plain", "selective"]


def items():
    return sorted(bodies())


__all__ = ["ARMS", "build", "detect", "items"]
