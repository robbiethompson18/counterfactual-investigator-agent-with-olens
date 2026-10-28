from gate import can_enter


def test_can_enter():
    assert can_enter("min", "admin,editor") is False
