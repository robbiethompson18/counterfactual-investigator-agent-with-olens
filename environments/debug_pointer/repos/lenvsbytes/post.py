from clean import clean
from encode import payload
from limits import within


def accepted(s, max_bytes):
    body = clean(s)
    payload(body)
    return within(body, max_bytes)
