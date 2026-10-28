from index import build
from query import lookup


def find(d, key):
    return lookup(build(d), key)
