CODES = {"SAVE15": 15, "SAVE5": 5}


def discount(code):
    return CODES.get(code, 0)
