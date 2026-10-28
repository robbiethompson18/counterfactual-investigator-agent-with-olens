from bounds import clamp
from config import MAX_LEVEL, MIN_LEVEL


def level(x):
    return clamp(MIN_LEVEL, x, MAX_LEVEL)
