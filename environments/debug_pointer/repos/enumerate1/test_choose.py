from choose import at_rank


def test_at_rank():
    assert at_rank(["c", "a", "b"], 1) == "a"
