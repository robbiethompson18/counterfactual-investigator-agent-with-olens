from money import amount


def total(xs):
    return sum(amount(x) for x in xs)
