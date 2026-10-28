from label import pct_label
from prefix import with_prefix
from ratio import hit_ratio


def cache_line(hits, total):
    return with_prefix(pct_label(hit_ratio(hits, total)))
