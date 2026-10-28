from envvars import get
from flags import flag
from names import DEBUG


def debug_on(env):
    return flag(get(env, DEBUG))
