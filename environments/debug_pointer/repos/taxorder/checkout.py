from discount import apply_discount
from money import cents
from tax import apply_tax


def total(price, off, tax_rate):
    return cents(apply_discount(apply_tax(price, tax_rate), off))
