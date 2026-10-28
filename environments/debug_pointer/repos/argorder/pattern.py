import re

USER = re.compile(r"^[a-z][a-z0-9_]*$")


def looks_like_user(s):
    return USER.match(s) is not None
