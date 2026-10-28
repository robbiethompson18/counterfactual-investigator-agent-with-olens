from guard import ordered
from show import fmt_seconds
from watch import elapsed_ms


def took(start_ms, end_ms):
    a, b = ordered(start_ms, end_ms)
    return fmt_seconds(elapsed_ms(a, b))
