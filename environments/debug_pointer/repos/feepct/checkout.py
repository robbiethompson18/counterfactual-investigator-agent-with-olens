from fee import fee
from quote import with_fee
from rates import rate


def due(amount, method):
    return with_fee(amount, fee(amount, rate(method)))
