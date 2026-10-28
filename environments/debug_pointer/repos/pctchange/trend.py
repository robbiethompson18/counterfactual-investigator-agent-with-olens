from change import change
from fmt import fmt_pct
from sign import signed


def trend(prev, cur):
    return fmt_pct(signed(change(prev, cur)))
