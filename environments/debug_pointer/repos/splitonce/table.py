from clean import clean
from kv import parse


def table(text):
    return {clean(k): clean(v) for k, v in (parse(line) for line in text.splitlines())}
