from match import same
from sums import total


def reconciled(xs, expected):
    return same(total(xs), expected)
