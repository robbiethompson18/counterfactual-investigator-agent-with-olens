from meter import bar


def test_bar():
    assert bar(150) == "#" * 10
