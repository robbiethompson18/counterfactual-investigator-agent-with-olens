def normalize(dx, dy):
    n = (dx * dx + dy * dy) ** 0.5
    return dx / n, dy / n
