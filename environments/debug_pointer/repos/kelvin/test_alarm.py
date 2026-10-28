from alarm import status


def test_status():
    assert status(263) == "frozen"
