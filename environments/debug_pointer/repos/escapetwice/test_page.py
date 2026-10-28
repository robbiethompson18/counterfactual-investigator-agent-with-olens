from page import page


def test_page():
    assert page("Tom & Jerry") == "<h1>Tom &amp; Jerry</h1>"
