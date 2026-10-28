from norm import norm


def build(d):
    return {norm(k): v for k, v in d.items()}
