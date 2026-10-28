def display(hour, offset):
    return f"{(hour + offset) % 24:02d}:00"
