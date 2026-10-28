from fmt import fmt_dollars
from wallet import balance_cents
from words import labelled


def line(entries):
    return labelled(fmt_dollars(balance_cents(entries)))
