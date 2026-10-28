def scores(text):
    out = {}
    for line in text.splitlines():
        name, s = line.split()
        out[name] = int(s)
    return out
