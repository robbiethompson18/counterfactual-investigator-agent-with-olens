"""Hack rates for variants.py runs and each variant vs the repo-default base, framed as changes that move hacking down
(base->variant when the variant hacks less, variant->base otherwise). Margin test as in margin.py: |Δ| > 10 pp at p < 0.01."""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from scipy.optimize import brentq
from scipy.stats import fisher_exact, norm


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


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from endkind import end_kind

runs = sys.argv[1:] or ["var_v4f", "var_noit_v4f"]
out = []
for run in runs:
    by = defaultdict(list)
    for f in (ROOT / "runs" / run).glob("*.json"):
        d = json.loads(Path(f).read_text())
        r = d["record"]
        r["end_kind"] = end_kind(r, d["messages"])
        by[r["cond"]].append(r)
    cost = sum(r["cost"] for rs in by.values() for r in rs)
    out += [
        f"## {run} ({sum(map(len, by.values()))} episodes, ${cost:.2f})",
        "",
        "| condition | n | hack % [95% CI] | ends | notes |",
        "|---|---|---|---|---|",
    ]
    for c, rs in sorted(by.items(), key=lambda kv: (kv[0] != "base", kv[0])):
        n, k = len(rs), sum(r["hack"] for r in rs)
        lo, hi = wilson(k, n)
        ends = defaultdict(int)
        for r in rs:
            ends[r["end_kind"]] += 1
        notes = []
        if c == "rw_self_report":
            notes.append(f"claimed --pass {sum(r['self_report_pass'] for r in rs)}/{n}")
        if c == "writable":
            notes.append(f"edited tests {sum(r['tests_modified'] for r in rs)}/{n}")
        notes.append(f"7 prime {sum(r['honest_7'] for r in rs)}/{n}")
        out.append(
            f"| {c} | {n} | {100 * k / n:.0f} [{100 * lo:.0f}, {100 * hi:.0f}] | "
            + ", ".join(f"{e} {v}" for e, v in sorted(ends.items()))
            + f" | {'; '.join(notes)} |"
        )
    b = by["base"]
    bk, bn = sum(r["hack"] for r in b), len(b)
    out += [
        "",
        "Single-factor diffs that lower hacking (|Δ| > 10 pp at p < 0.01 marked **down**):",
        "",
        "| from → to | hack % | Δ pp | Fisher p | margin p | |",
        "|---|---|---|---|---|---|",
    ]
    rows = []
    for c, rs in by.items():
        if c == "base":
            continue
        k, n = sum(r["hack"] for r in rs), len(rs)
        (hk, hn, hname), (lk, ln, lname) = sorted([(bk, bn, "base"), (k, n, c)], key=lambda t: -t[0] / t[1])
        d = 100 * (lk / ln - hk / hn)
        pm = fm_p(lk, ln, hk, hn, 0.10)  # H1: high - low > 10 pp
        pf = fisher_exact([[hk, hn - hk], [lk, ln - lk]])[1]
        rows.append(
            (
                d,
                f"| {hname} → {lname} | {100 * hk / hn:.0f} → {100 * lk / ln:.0f} | {d:+.0f} | {pf:.1g} | {pm:.1g} | {'**down**' if pm < 0.01 else ''} |",  # noqa: E501
            )
        )
    out += [r for _, r in sorted(rows)] + [""]
print("\n".join(out))
