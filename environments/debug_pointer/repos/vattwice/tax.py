VAT = 0.2


def add_vat(net):
    return round(net * (1 + VAT), 2)
