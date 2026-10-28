from blank import drop_blank
from count import count
from load import rows


def n_rows(text):
    return count(drop_blank(rows(text)))
