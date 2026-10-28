from tag import tag


def test_tag():
    assert tag("tea") == "tea: $1.50"
