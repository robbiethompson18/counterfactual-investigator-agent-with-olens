from default import or_default
from ids import parse_id
from lookup import find, table


def label_for(raw, pairs):
    return or_default(find(table(pairs), parse_id(raw)))
