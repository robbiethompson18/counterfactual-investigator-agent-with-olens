from locate import where


def test_where():
    assert where("a\nb\nTODO\nc", "TODO") == "line 3"
