def allowed(role, csv):
    return role in ",".join(r for r in csv.split(",") if r)
