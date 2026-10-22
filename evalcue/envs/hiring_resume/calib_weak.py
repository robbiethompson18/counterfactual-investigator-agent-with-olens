"""Calibration pass over the weaker resume batch (ids like swe_w01; 8 jobs x 12) after the first batch came out ~98% YES.
Run with --n 192 for 2 samples per body per arm. See hiring_resume/calib.py.
"""

from evalcue.envs.hiring_resume.data.core import build, detect, interleaved

ARMS = ["base", "selective", "bar_half"]


def items():
    return interleaved(weak=True)


__all__ = ["ARMS", "build", "detect", "items"]
