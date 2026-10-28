from ledger import total


def test_total():
    assert total("name,amount\na,10\nb,20\nc,30", ["x", "y"]) == 60
