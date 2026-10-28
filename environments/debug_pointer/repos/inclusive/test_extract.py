from extract import between


def test_between():
    assert between("abcdef", "b", "e") == "bcde"
