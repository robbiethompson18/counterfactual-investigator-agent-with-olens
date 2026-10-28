from snippet import snippet


def test_snippet():
    assert snippet("hello world", "w") == "world"
