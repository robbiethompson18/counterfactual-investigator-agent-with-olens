from indent import depths


def test_depths():
    assert depths("a\n  b\n    c") == [0, 2, 4]
