"""Posteriors and the Pool's scores.

Each arm's rate r ~ Beta(1+k, 1+n-k). Forecaster correct-pick rate p ~ Beta(1+k, 1+n-k) too.
    L   = 2.5th percentile of D = r_null - r_real                       (largeness, well measured)
    P+  = P(D > 0)                                                       (eligibility needs >= 0.8)
    U   = 1 - E|2p - 1|                                                  (forecaster near chance)
    U'  = E[1 - p]                                                       (forecaster wrong counts as surprising)
    N   = P(|r_null - r_base| <= 0.10)                                   (the null behaves like a null)
Monte Carlo with a fixed seed so scores are reproducible.
"""

import random

DRAWS = 10_000
NULL_TOL = 0.10


def _beta(k, n, rng):
    return [rng.betavariate(1 + k, 1 + n - k) for _ in range(DRAWS)]


def quantile(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, int(q * len(s))))]


def interval(xs, mass=0.8):
    return [round(quantile(xs, (1 - mass) / 2), 4), round(quantile(xs, 1 - (1 - mass) / 2), 4)]


def scores(rates: dict, forecaster: tuple[int, int] | None) -> dict:
    """rates: {"baseline": (k, n), "real": (k, n), "null": (k, n)}; forecaster: (correct, trials)."""
    rng = random.Random(0)
    r = {a: _beta(*rates[a], rng) for a in ("baseline", "real", "null")}
    d = [x - y for x, y in zip(r["null"], r["real"])]
    dn = [x - y for x, y in zip(r["null"], r["baseline"])]
    out = {
        "L": round(quantile(d, 0.025), 4),
        "P_pos": round(sum(x > 0 for x in d) / DRAWS, 4),
        "N": round(sum(abs(x) <= NULL_TOL for x in dn) / DRAWS, 4),
        "intervals80": {**{a: interval(r[a]) for a in r}, "delta": interval(d), "null_minus_base": interval(dn)},
    }
    if forecaster and forecaster[1]:
        p = _beta(*forecaster, rng)
        out["U"] = round(1 - sum(abs(2 * x - 1) for x in p) / DRAWS, 4)
        out["U_rev"] = round(1 - sum(p) / DRAWS, 4)
        out["intervals80"]["picker_p"] = interval(p)
    else:  # unmeasured: the flat-prior values, so the node is neither rewarded nor punished
        out["U"], out["U_rev"] = 0.5, 0.5
    return out
