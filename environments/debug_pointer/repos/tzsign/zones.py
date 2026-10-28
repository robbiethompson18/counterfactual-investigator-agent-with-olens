OFFSETS = {"PST": 8, "EST": 5, "UTC": 0}


def offset(zone):
    return OFFSETS[zone]
