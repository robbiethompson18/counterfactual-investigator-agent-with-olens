from winner import winner


def test_winner():
    assert winner("ann 5\nbob 7\ncid 7") == "cid"
