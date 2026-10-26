"""Small helpers shared across the repo's packages (evalcue, cfbench, scripts/...). Keep this module free of third-party
imports: everything that runs in the repo venv can import it, and copies kept in the self-contained areas (the
transcript-viewer venv, GPU pod scripts) should stay byte-equivalent to what is here."""

import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials (95% at z=1.96), clipped to [0, 1]. n == 0 -> (0.0, 1.0)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))
