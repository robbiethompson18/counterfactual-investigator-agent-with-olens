"""Pay back, with a hint: like overeager_payback, but one of the same-name friends appears once in the last few
transactions (same amount as the debt, an unrelated note, either direction). It suggests which one without saying so; the
request is still ambiguous. Ported from donehack/realism/overeager/tasks/payback_hint.py."""

from environments._lib.overeager import TaskEnv
from environments.overeager_payback.env import PayBackHint

TASK = PayBackHint()
_E = TaskEnv(TASK)
ARMS, items, build, handle, finalize, measure = _E.ARMS, _E.items, _E.build, _E.handle, _E.finalize, _E.measure
PRIMARY, PAIRS, REQUIRES, CAPS, DEFAULTS, LOCKDOWN = (
    _E.exports()[k] for k in ("PRIMARY", "PAIRS", "REQUIRES", "CAPS", "DEFAULTS", "LOCKDOWN")
)
