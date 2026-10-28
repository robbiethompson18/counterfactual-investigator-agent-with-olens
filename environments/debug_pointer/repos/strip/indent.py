from depth import depth
from read import lines


def depths(text):
    return [depth(line) for line in lines(text)]
