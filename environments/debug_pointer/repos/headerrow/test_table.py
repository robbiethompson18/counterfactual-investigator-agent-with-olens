from table import n_rows


def test_n_rows():
    assert n_rows("id,name\n1,a\n2,b\n3,c") == 3
