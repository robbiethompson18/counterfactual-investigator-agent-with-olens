from fetch import field
from render import render


def show(record, key):
    return render(field(record, key))
