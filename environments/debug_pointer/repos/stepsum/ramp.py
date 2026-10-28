from steps import gaps


def climb(n_steps, rise):
    total = 0
    for _ in range(gaps(n_steps)):
        total += rise
    return total
