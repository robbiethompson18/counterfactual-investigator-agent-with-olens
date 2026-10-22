"""Hiring decisions from one borderline resume: prompt-context arms (a plain posting instead of Karvonen & Marks 2025
style company culture text, culture text with its diversity paragraph removed, their anti-bias statement, an eval cue, a
deployment cue), each crossed with White and Black, female and male names, so the context x name interaction is measurable.
Positive = DECISION: YES. See docs/envs/hiring_resume.md.
"""

from evalcue.envs.hiring_resume.attrs import BEHAVIOR  # noqa: F401  (re-exported for cfbench)
from evalcue.envs.hiring_resume.data.core import build, detect, interleaved

# The bare-context cells (culture text) are the same items x names as in hiring_resume_names; selectivity drops the weak
# batch to ~12% YES, so the selective cells were left out for time (see the doc). Opt in with --arms.
CONTEXTS = ["ctx_plain", "ctx_nodiv", "antibias", "eval", "deploy"]
NAMES = ["name_white_f", "name_white_m", "name_black_f", "name_black_m"]
ARMS = [f"{n}+{c}" for c in CONTEXTS for n in NAMES]


def items():
    return interleaved(weak=True)  # the weak batch: ~63% YES under the default context


__all__ = ["ARMS", "build", "detect", "items"]
