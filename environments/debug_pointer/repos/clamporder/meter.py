from scale import level


def bar(x):
    return "#" * (level(x) // 10)
