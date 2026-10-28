from check import plausible
from person import age


def age_field(dob, today):
    a = age(dob, today)
    return a if plausible(a) else None
