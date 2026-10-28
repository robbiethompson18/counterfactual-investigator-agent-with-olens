from invoice import total


def test_total():
    assert total(["book"]) == "$10.80"
