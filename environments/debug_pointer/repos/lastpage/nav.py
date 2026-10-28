from listing import page_count
from size import page_size


def last_page(n, size=None):
    return page_count(n, page_size(size))
