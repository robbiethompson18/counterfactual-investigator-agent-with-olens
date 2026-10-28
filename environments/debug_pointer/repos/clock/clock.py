from fmt import hhmm
from offset import shift
from split import parts


def wall(seconds, tz_hours=0):
    h, m = parts(shift(seconds, tz_hours))
    return hhmm(h, m)
