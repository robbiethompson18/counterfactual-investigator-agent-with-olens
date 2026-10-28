def entries(rows):
    return [(-amt if kind == "debit" else amt) for kind, amt in rows]
