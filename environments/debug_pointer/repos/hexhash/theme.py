from css import color_rule
from lower import lower
from palette import color


def rule(name):
    return color_rule(lower(color(name)))
