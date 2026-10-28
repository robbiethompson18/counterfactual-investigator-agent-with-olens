from summary import average


def test_average():
    assert average([4, None, 8]) == 6.0
