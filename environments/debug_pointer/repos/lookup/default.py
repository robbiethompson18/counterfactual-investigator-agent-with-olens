MISSING = "unknown"


def or_default(value):
    return value if value is not None else MISSING
