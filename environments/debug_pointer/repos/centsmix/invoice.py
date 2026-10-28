from money import dollars
from price import price
from tax import with_tax


def total(skus):
    return dollars(with_tax(sum(price(s) for s in skus)))
