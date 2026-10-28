from theme import rule


def test_rule():
    assert rule("alert") == "color: #ff0000;"
