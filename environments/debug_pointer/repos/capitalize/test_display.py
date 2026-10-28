from display import display


def test_display():
    assert display("  McDonald ") == "McDonald"
