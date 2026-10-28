from size import size


def test_size():
    assert size(3072) == "3.0 KiB"
