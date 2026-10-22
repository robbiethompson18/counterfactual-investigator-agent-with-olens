"""hiring_resume_names over ALL 144 drafted bodies (not just the calibrated borderline ones), one sample per body per arm at
--n 144. Launched before calibration finished, so the queue position wasn't wasted; analyze it restricted to the bodies that
the independent calibration run (hiring_resume_calib) found borderline, plus on all bodies. Positive = DECISION: YES.
"""

from evalcue.envs.hiring_resume_data.core import bodies, build, detect
from evalcue.envs.hiring_resume_names import ARMS


def items():
    return sorted(bodies())


__all__ = ["ARMS", "build", "detect", "items"]
