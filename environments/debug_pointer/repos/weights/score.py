from weights import weight_of


def score(name, raw, weights):
    return raw * weight_of(name, weights)
