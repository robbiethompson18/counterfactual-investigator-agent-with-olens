from slot import slot


def test_slot():
    assert slot(22, 2) == "22:00-00:00"
