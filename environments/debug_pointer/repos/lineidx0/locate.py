from label import line_label
from scan import find_idx0
from split import lines


def where(text, needle):
    return line_label(find_idx0(lines(text), needle))
