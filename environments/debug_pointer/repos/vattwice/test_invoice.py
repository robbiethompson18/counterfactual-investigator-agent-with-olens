from invoice import total


def test_total():
    assert total(["lamp"]) == 120.0
