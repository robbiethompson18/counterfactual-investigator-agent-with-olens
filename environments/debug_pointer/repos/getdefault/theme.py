from merge import merged
from prefs import pref


def theme(user):
    return pref(merged(user), "theme")
