def trim(s, n=60):
    return s if len(s) <= n else s[: n - 1] + "..."
