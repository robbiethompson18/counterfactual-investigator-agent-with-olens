from trend import trend


def test_trend():
    assert trend(200, 250) == "+25%"
