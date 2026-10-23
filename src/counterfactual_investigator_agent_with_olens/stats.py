"""Rate intervals and tests shared by the env screens and evalcue. Stdlib only."""

import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n; (0, 1) when n == 0."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def fisher_greater(k_a: int, n_a: int, k_b: int, n_b: int) -> float:
    """One-sided Fisher exact p for rate_A > rate_B: P(X >= k_a) under the hypergeometric null."""
    total, k = n_a + n_b, k_a + k_b
    denom = math.comb(total, n_a)
    return sum(math.comb(k, x) * math.comb(total - k, n_a - x) for x in range(k_a, min(k, n_a) + 1)) / denom
