from rank import ranked
from score import score


def best(raw, weights):
    scores = {n: score(n, v, weights) for n, v in raw.items()}
    return ranked(scores)[0][0]
