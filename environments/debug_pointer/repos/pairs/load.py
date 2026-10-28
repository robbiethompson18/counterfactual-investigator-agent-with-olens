def amounts(text):
    return [int(line.split(",")[1]) for line in text.splitlines()[1:]]
