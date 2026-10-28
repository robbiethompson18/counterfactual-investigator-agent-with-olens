from checkout import pay


def test_pay():
    assert pay(40.0, "SAVE15") == 34.0
