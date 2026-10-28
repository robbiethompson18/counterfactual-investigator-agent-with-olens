from fmt import dollars
from names import label
from price import price


def tag(name):
    return f"{label(name)}: {dollars(price(name))}"
