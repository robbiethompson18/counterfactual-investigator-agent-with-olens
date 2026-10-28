def ids(text):
    return [line.strip().lstrip("0") or "0" for line in text.splitlines()]
