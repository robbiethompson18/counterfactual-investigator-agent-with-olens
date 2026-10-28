from defaults import PLACEHOLDER


def render(value):
    return PLACEHOLDER if value is None else str(value)
