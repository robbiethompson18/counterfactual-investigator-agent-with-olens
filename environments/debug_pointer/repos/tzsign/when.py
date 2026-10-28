from local import to_local
from zones import offset


def local_hour(utc_hour, zone):
    return to_local(utc_hour, offset(zone))
