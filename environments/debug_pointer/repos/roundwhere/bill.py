from line import line_total
from sums import add


def total(prices):
    return round(add(line_total(p) for p in prices), 2)
