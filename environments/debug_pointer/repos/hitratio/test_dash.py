from dash import cache_line


def test_cache_line():
    assert cache_line(25, 100) == "hits: 25%"
