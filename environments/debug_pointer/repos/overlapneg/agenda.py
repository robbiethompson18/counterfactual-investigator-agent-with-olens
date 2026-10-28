from busy import busy_minutes
from sortslots import by_start


def load(slots, window):
    return busy_minutes(by_start(slots), window)
