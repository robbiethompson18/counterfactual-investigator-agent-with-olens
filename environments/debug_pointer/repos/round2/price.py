from conv import dollars
from fmt import money
from sign import signed


def display(cents):
    return money(dollars(signed(cents)))
