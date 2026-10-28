from day import weekday
from names import name
from wrap import wrap


def day_name(y, m, d):
    return name(wrap(weekday(y, m, d)))
