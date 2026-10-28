from length import ok_length
from pattern import looks_like_user


def valid(s):
    return looks_like_user(s) and ok_length(s, 4)
