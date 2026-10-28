from find import find


def test_find():
    assert find({"Ada": 1, "Bob": 2}, "Ada") == 1
