from conf import value


def test_value():
    assert value("home: http://x.io/a", "home") == "http://x.io/a"
