import datetime

from person import age


def test_age():
    assert age(datetime.date(2000, 6, 15), datetime.date(2024, 3, 1)) == 23
