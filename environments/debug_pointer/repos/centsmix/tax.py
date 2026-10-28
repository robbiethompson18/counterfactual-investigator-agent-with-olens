RATE = 0.08


def with_tax(amount):
    return round(amount * (1 + RATE), 2)
