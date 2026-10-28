from price import cost
from rental import hire_days


def charge(pickup, dropoff, daily):
    return cost(hire_days(pickup, dropoff), daily)
