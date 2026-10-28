def shift(seconds, tz_hours):
    return (seconds + tz_hours * 3600) % 86400
