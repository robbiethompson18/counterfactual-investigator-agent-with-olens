from busy import busy_minutes


def test_busy_minutes():
    assert busy_minutes([(0, 30), (100, 130)], (20, 60)) == 10
