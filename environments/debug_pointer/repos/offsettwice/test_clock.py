from clock import show


def test_show():
    assert show(12, 2) == "14:00"
