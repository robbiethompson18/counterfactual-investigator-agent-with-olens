from fmt import one_decimal
from scrub import scrub
from stats import mean


def average(xs):
    return one_decimal(mean(scrub(xs)))
