from load import scores
from names import clean
from pick import best


def winner(text):
    return clean(best(scores(text)))
