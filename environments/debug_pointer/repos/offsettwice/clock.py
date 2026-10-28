from display import display
from localtime import to_local
from zone import offset_for


def show(utc_hour, zone_hours):
    off = offset_for(zone_hours)
    return display(to_local(utc_hour, off), off)
