from release import pick


def test_pick():
    assert pick(["v1.9.0", "v1.10.0"]) == "1.10.0"
