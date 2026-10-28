from trip import summary


def test_summary():
    assert summary([(0, 0), (0, 300), (400, 300)]) == "0.7 km"
