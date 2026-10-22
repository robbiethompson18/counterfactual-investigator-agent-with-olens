"""Karvonen & Marks's regime: name arms (as in hiring_resume_names) with the top-10% selectivity text, on the FIRST (strong)
resume batch, which is ~98% YES with the default context but ~66% YES with selectivity. Items are all 144 first-batch bodies,
interleaved across jobs. Positive = DECISION: YES. See docs/envs/hiring_resume.md.
"""

from evalcue.envs.hiring_resume.attrs import BEHAVIOR  # noqa: F401  (re-exported for cfbench)
from evalcue.envs.hiring_resume.data.core import NAME_ARMS, build, detect, interleaved

ARMS = ["selective", "name_none+selective"] + [f"{a}+selective" for a in NAME_ARMS]


def items():
    return interleaved(weak=False)


__all__ = ["ARMS", "build", "detect", "items"]
