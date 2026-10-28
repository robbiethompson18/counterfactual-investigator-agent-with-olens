from catalog import unit_price
from lines import items
from qty import count


def total(text):
    return sum(unit_price(name) * count(q) for name, q in items(text))
