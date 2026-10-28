POINTS = ["N", "E", "S", "W"]


def compass(deg):
    return POINTS[int(((deg % 360) + 45) // 90) % 4]
