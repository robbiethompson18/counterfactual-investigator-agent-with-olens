from load import amounts
from pair import labelled
from sums import add_up


def total(text, labels):
    return add_up(labelled(labels, amounts(text)))
