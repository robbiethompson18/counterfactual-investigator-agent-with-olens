from axis import axis
from text import text


def labels(lo, hi, intervals):
    return [text(t) for t in axis(lo, hi, intervals)]
