def items(text):
    return [line.strip().split(":") for line in text.strip().splitlines()]
