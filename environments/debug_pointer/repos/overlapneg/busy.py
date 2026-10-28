from overlap import overlap


def busy_minutes(slots, window):
    w0, w1 = window
    return sum(overlap(s0, s1, w0, w1) for s0, s1 in slots)
