from base import base
from join import join
from safe import safe


def url(name):
    return join(base(), safe(name))
