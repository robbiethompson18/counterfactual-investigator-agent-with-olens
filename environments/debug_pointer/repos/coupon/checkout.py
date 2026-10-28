from cart import apply_discount
from coupon import discount
from floor import floor_zero


def pay(price, code):
    return floor_zero(apply_discount(price, discount(code)))
