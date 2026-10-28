from safe import escape
from template import h1


def render(title):
    return h1(escape(title))
