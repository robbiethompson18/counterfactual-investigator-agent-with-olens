from limit import over_limit


def test_over_limit():
    assert over_limit(100.0, 1.0, "motorway") is False
