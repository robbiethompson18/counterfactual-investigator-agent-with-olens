def to_local(utc_hour, offset):
    return (utc_hour + offset) % 24
