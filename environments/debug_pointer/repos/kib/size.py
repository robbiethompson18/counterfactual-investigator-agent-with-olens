from clamp import nonneg
from human import human
from units import to_k


def size(n):
    return human(to_k(nonneg(n)))
