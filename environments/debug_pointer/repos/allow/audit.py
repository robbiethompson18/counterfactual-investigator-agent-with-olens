LOG = []


def note(role, ok):
    LOG.append((role, ok))
    return ok
