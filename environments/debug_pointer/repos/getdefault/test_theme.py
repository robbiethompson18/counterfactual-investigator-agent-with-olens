from theme import theme


def test_theme():
    assert theme({"theme": None}) == "light"
