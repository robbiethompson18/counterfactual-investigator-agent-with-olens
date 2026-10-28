from rate import taxed


def line_total(x):
    return round(taxed(x), 2)
