from latest import latest


def test_latest():
    assert latest("02/03/2024\n31/12/2024\n05/01/2025") == "05/01/2025"
