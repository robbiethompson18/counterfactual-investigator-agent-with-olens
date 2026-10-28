from report import headline


def test_headline():
    assert headline([1, 2, 3, 100]) == "typical: 2.5"
