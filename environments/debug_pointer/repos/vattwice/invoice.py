from lines import subtotal
from prices import price
from tax import add_vat


def total(skus):
    return add_vat(subtotal(price(s) for s in skus))
