from geo import dist


def total_m(points):
    return sum(dist(points[i], points[i + 1]) for i in range(len(points) - 1))
