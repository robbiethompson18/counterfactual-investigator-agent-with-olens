"""hiring_resume_pairwise name arms on only the borderline pairs: the pairs whose P(A), pooled over all 25 arms of
hiring_resume_pairwise/2026-10-21_r3_all_arms_n30, was in [0.05, 0.95] (most pairs were 0 or 1 in every arm). Fresh samples.
Items: hiring_resume/data/pairwise_borderline.json. Positive = Qwen picks A.
"""

import json
from pathlib import Path

from evalcue.envs.hiring_resume.pairwise import ARMS as ALL_ARMS
from evalcue.envs.hiring_resume.pairwise import (
    BEHAVIOR,  # noqa: F401  (re-exported for cfbench)
    build,
    detect,
)

ARMS = [a for a in ALL_ARMS if a == "none~none" or a.startswith("name_") or "visa_h1b" in a]


def items():
    return json.loads((Path(__file__).parent / "data" / "pairwise_borderline.json").read_text())["items"]


__all__ = ["ARMS", "build", "detect", "items"]
