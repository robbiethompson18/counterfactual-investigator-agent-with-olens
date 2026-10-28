from rules import valid


def register(name, taken):
    if name in taken or not valid(name):
        return "rejected"
    return "ok"
