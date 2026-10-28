from allow import allowed
from audit import note


def can_enter(role, csv):
    return note(role, allowed(role, csv))
