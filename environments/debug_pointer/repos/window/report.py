from load import rows
from parse import values
from window import moving_max


def peaks(text, k):
    return moving_max(values(rows(text)), k)
