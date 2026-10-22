"""Pairs where the hack-rate change exceeds 10 pp at one-sided p < 0.01 (H0: |Δ| <= 10 pp).
Farrington–Manning score test for a difference of proportions against a margin."""

import csv
import math
from pathlib import Path

from scipy.optimize import brentq
from scipy.stats import norm


def fm_p(k1, n1, k2, n2, margin):
    """One-sided p for H0: p2 - p1 <= margin (H1: p2 - p1 > margin)."""
    x1, x2 = k1 / n1, k2 / n2
    d = x2 - x1
    # restricted MLE of p1 under p2 = p1 + margin
    lo, hi = max(0, -margin) + 1e-12, min(1, 1 - margin) - 1e-12

    def score(p1):
        p2 = p1 + margin
        return k1 / p1 - (n1 - k1) / (1 - p1) + k2 / p2 - (n2 - k2) / (1 - p2)

    try:
        p1 = brentq(score, lo, hi)
    except ValueError:
        p1 = min(max(x1, lo), hi)
    p2 = p1 + margin
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return norm.sf((d - margin) / se)


rows = list(csv.DictReader(Path("pairs.csv").read_text().splitlines()))
out = []
for r in rows:
    k1, n1, k2, n2 = int(r["base_k"]), int(r["base_n"]), int(r["var_k"]), int(r["var_n"])
    up = fm_p(k1, n1, k2, n2, 0.10)  # Δ > +10
    down = fm_p(k2, n2, k1, n1, 0.10)  # Δ < -10
    r["p_margin"] = min(up, down)
    out.append(r)
with open("pairs_margin.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
for model in ["V4 Flash", "Qwen3.6-27B"]:
    rs = [r for r in out if r["model"] == model]
    sig = [r for r in rs if r["p_margin"] < 0.01]
    one = [r for r in sig if r["n_factors"] == "1"]
    print(f"== {model}: {len(sig)}/{len(rs)} pairs with Δ>10pp at p<0.01; single-factor: {len(one)}")
    for r in sorted(one, key=lambda r: -abs(float(r["delta_pp"]))):
        print(
            f"   {r['base']:22s} -> {r['variant']:22s} {float(r['base_rate']):3.0f}% -> {float(r['var_rate']):3.0f}%  Δ{float(r['delta_pp']):+5.0f}  p={r['p_margin']:.1g}"  # noqa: E501
        )
