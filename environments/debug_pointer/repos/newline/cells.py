from fields import fields
from pick import at
from reader import lines


def cell(text, r, c):
    return at([fields(line) for line in lines(text)], r, c)
