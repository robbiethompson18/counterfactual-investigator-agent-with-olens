from caps import headline_case
from clean import clean
from trim import trim


def headline(raw):
    return trim(headline_case(clean(raw)))
