def nth(xs, rank):
    for i, _ in enumerate(xs, 1):
        if i == rank:
            return xs[i]
    return None
