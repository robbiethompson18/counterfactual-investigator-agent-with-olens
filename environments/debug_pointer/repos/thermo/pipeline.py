from convert import c_to_f
from display import show
from sensor import celsius


def reading_f(raw):
    return show(c_to_f(celsius(raw)))
