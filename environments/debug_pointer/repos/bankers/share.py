from fmt import pct_str
from guard import safe_div
from pct import pct


def share(a, b):
    return pct_str(pct(*safe_div(a, b)))
