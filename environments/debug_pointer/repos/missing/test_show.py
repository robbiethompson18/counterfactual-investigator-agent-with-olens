from show import show


def test_show():
    assert show({"name": "Ada"}, "email") == "n/a"
