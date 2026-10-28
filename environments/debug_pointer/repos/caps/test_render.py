from render import headline


def test_headline():
    assert headline("  the ART of war ") == "The Art of War"
