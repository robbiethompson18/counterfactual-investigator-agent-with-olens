from ident import ident
from names import drop_ext
from paths import basename


def module_name(path):
    return ident(drop_ext(basename(path)))
