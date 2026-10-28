from bill import whole_dollars


def test_whole_dollars():
    assert whole_dollars([1.7, 2.9]) == 5
