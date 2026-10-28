from receipt import line


def test_line():
    assert line([250, 199]) == "balance $4.49"
