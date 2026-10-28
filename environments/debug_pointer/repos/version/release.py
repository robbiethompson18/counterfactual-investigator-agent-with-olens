from newest import latest
from tags import strip_v


def pick(tags):
    return latest([strip_v(t) for t in tags])
