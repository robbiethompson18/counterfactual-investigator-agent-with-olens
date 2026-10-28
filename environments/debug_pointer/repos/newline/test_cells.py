from cells import cell


def test_cell():
    assert cell("a,b\nc,d\n", 1, 1) == "d"
