from resolve import url


def test_url():
    assert url("img.png") == "https://cdn.example.com/static/img.png"
