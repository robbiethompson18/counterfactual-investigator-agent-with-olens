from check import nonneg
from human import fmt_kb
from store import size_bytes


def usage(blobs):
    return fmt_kb(nonneg(size_bytes(blobs)))
