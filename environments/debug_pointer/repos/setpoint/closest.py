from delta import delta
from first import first_by
from valid import valid


def closest_reading(target, xs):
    return first_by([(x, delta(x, target)) for x in valid(xs)])
