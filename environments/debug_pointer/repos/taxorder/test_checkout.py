from checkout import total


def test_total():
    assert total(100.0, 10.0, 0.2) == 108.0
