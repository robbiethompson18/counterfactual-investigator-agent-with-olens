from plot import labels


def test_labels():
    assert labels(0, 10, 5) == ["0", "2", "4", "6", "8", "10"]
