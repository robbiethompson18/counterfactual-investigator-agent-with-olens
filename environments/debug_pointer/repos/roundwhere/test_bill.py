from bill import total


def test_total():
    assert total([1.004, 2.004, 3.004]) == 6.01
