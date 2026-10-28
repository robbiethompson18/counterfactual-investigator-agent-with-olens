def to_local(utc_hour, offset_hours):
    return (utc_hour + offset_hours) % 24
