def moving_max(xs, k):
    out = []
    for i in range(len(xs) - k):
        out.append(max(xs[i : i + k]))
    return out
