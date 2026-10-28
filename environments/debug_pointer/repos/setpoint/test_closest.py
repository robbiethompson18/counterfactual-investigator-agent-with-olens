from closest import closest_reading


def test_closest_reading():
    assert closest_reading(20, [15, 22, 40]) == 22
