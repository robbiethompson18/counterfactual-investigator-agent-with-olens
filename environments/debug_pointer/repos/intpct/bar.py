from fmt import fmt
from guard import safe
from pct import pct


def bar(a, b):
    return fmt(pct(*safe(a, b)))
