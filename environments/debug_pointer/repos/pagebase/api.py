from offset import start
from pages import page_number
from window import window


def page(xs, raw_page, size):
    return window(xs, start(page_number(raw_page), size), size)
