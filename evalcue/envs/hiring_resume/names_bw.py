"""Replication of the borderline pro-Black gap with every name forced: each of the 6 names in the White and Black, female and
male cells is its own arm (name_<group>_<f|m>_<k>), on the 11 borderline weak bodies (calibrated.json). Group effect = pooled
Black arms vs pooled White arms across 12 names each; the spread between names within a cell (e.g. Emily Walsh vs Anne Murphy)
is the name-specific placebo. Run with --n 22 (2 per body). Positive = DECISION: YES.
"""

from evalcue.envs.hiring_resume.attrs import BEHAVIOR  # noqa: F401  (re-exported for cfbench)
from evalcue.envs.hiring_resume.data.core import build, calibrated, detect

ARMS = ["base"] + [f"name_{g}_{s}_{k}" for g in ("white", "black") for s in "fm" for k in range(6)]


def items():
    return calibrated()


__all__ = ["ARMS", "build", "detect", "items"]
