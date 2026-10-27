"""EvilGenie (arXiv 2511.21654): hard LiveCodeBench problems with hackable public tests (the agent can hardcode outputs),
2 problems in our set. Same task code as overeager_impossiblebench."""

from environments._lib.overeager import TaskEnv
from environments.overeager_impossiblebench.env import Coding

TASK = Coding("evilgenie", "eg:")
_E = TaskEnv(TASK)  # items are indices mod the problem count, as the old runs used them
ARMS, items, build, handle, finalize, measure = _E.ARMS, _E.items, _E.build, _E.handle, _E.finalize, _E.measure
PRIMARY, PAIRS, REQUIRES, CAPS, DEFAULTS, LOCKDOWN = (
    _E.exports()[k] for k in ("PRIMARY", "PAIRS", "REQUIRES", "CAPS", "DEFAULTS", "LOCKDOWN")
)
