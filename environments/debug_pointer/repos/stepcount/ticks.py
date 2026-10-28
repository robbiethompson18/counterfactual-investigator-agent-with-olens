def ticks(lo, hi, n):
    step = (hi - lo) / n
    return [lo + i * step for i in range(n)]
