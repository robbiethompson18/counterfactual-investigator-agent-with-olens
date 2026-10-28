from caps import cap
from clean import clean
from width import fit


def display(s):
    return fit(cap(clean(s)))
