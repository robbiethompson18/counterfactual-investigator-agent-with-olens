from checkout import due


def test_due():
    assert due(200, "card") == 220.0
