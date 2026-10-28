from bounds import end_inclusive
from cut import cut
from start import start_of


def snippet(s, ch):
    a = start_of(s, ch)
    return cut(s, a, end_inclusive(s, a))
