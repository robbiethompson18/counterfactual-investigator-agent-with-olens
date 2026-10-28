def drop_blank(rows):
    return [r for r in rows if any(c.strip() for c in r)]
