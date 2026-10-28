from total import total


def test_total():
    assert total("apple:2\npear:1\n") == 1.75
