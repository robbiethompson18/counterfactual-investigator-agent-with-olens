from report import usage


def test_usage():
    assert usage([2048, 1024]) == "3.0 KB"
