from apply import apply
from check import nonneg
from ledger import entries


def balance(start, rows):
    total = start
    for amt in entries(rows):
        total = apply(total, amt)
    return nonneg(total)
