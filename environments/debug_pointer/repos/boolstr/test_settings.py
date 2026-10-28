from settings import debug_on


def test_debug_on():
    assert debug_on({"DEBUG": "false"}) is False
