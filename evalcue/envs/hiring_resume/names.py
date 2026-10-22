"""Hiring decisions from one borderline resume, name swapped: the gender {unclear, f, m} x race {unclear + 7 groups} grid,
one name per cell (data/core.py NAMES; name_x_x is the default name, i.e. a replicate of base), plus a redacted name.
Default context (company culture text). Positive = DECISION: YES. See docs/envs/hiring_resume.md and hiring_resume/data/core.py.
"""

from evalcue.envs.hiring_resume.attrs import BEHAVIOR  # noqa: F401  (re-exported for cfbench)
from evalcue.envs.hiring_resume.data.core import NAME_ARMS, build, detect, interleaved

ARMS = ["base", "name_none", *NAME_ARMS]


def items():
    return interleaved(weak=True)  # the weak batch: ~63% YES under the default context


__all__ = ["ARMS", "build", "detect", "items"]
