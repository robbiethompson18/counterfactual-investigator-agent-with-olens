from loader import dates
from order import by_date
from tail import last


def latest(text):
    return last(by_date(dates(text)))
