from api import page


def test_page():
    assert page(list(range(1, 26)), 1, 10) == list(range(1, 11))
