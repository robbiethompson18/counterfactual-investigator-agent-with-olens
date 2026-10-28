from margin import with_margin
from rules import limit_mph
from speedo import speed_kph


def over_limit(km, hours, road):
    return speed_kph(km, hours) > with_margin(limit_mph(road))
