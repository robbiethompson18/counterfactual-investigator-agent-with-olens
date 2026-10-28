from top import best


def test_best():
    assert best({"a": 5, "b": 2}, {"b": 2}) == "b"
