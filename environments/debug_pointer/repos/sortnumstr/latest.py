from ids import ids
from order import ascending
from tail import last


def latest_id(text):
    return last(ascending(ids(text)))
