from latest import latest_id


def test_latest_id():
    assert latest_id("9\n10\n2") == "10"
