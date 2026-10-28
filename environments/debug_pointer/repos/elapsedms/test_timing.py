from timing import took


def test_took():
    assert took(1000, 2500) == "1.5s"
