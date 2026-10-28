from freeze import is_freezing
from probe import read_kelvin
from words import word


def status(raw):
    return word(is_freezing(read_kelvin(raw)))
