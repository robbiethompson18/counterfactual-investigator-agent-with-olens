from load import numbers
from robust import typical


def headline(xs):
    return f"typical: {typical(numbers(xs)):g}"
