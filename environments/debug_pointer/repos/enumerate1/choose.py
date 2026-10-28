from order import ordered
from pick import nth


def at_rank(xs, rank):
    return nth(ordered(xs), rank)
