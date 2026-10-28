from balance import balance


def test_balance():
    assert balance(100, [("debit", 30), ("credit", 10)]) == 80
