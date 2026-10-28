PRICES = {"apple": 0.5, "pear": 0.75}


def unit_price(name):
    return PRICES.get(name.strip().title(), 0)
