from parts import numbers


def latest(vers):
    return max(vers, key=numbers)
