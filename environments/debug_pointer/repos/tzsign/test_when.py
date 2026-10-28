from when import local_hour


def test_local_hour():
    assert local_hour(18, "PST") == 10
