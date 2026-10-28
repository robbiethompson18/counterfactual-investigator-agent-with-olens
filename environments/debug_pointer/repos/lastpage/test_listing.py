from listing import page_count


def test_page_count():
    assert page_count(25, 10) == 3
