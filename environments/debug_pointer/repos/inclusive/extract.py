from check import nonempty
from cut import cut
from span import span


def between(s, first, last):
    a, b = span(s, first, last)
    return nonempty(cut(s, a, b))
