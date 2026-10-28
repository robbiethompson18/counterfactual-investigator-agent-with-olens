from report import peaks


def test_peaks():
    text = "day,qty\na,1\nb,3\nc,2\nd,5"
    assert peaks(text, 2) == [3, 3, 5]
