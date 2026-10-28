from compass import compass
from gyro import heading_rad
from unit import normalize


def direction(dx, dy):
    return compass(heading_rad(*normalize(dx, dy)))
