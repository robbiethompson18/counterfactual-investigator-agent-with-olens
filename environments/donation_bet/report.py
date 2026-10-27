"""Leakage gaps per run folder: P(at or below the line | good cause below) - P(at or below | good cause above), for each
other side (bad / beers / nothing) and lines set, as the mean over questions of the per-question gap (the estimand in
docs/envs/donation_bet.md, 95% CI from resampling within question) and pooled (95% Newcombe CI), plus bet_neutral's rate. Reads the latest
`measure` labels of each local run (data/envs/donation_bet/<run>/); unscorable replies are dropped.

    uv run python -m environments.donation_bet.report [RUN ...]      # default: every ref-template run
"""

import json
import math
import sys
from collections import defaultdict

import numpy as np

from environments._lib.index import latest_labels
from environments._lib.runfolder import LOCAL, read_episodes
from environments.donation_bet import env as E


def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def newcombe(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    (l1, u1), (l2, u2) = wilson(k1, n1), wilson(k2, n2)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def counts(run) -> dict[str, dict[str, list]]:
    """arm -> item -> list of 0/1 (at or below the line)."""
    _, labels = latest_labels(run)
    c: dict = defaultdict(lambda: defaultdict(list))
    for ep in read_episodes(run):
        v = (labels.get(ep["id"]) or {}).get("positive")
        if not ep.get("error") and v is not None:
            c[ep["arm"]][ep["item"]].append(int(v))
    return c


def kn(by_item) -> list[int]:
    return [sum(map(sum, by_item.values())), sum(map(len, by_item.values()))]


def qmean_gap(a, b, boot=2000, seed=0) -> tuple[float, float, float]:
    """Mean over the questions both arms have of P(a) - P(b), 95% CI resampling episodes within question."""
    rng = np.random.default_rng(seed)
    qs = sorted(set(a) & set(b))
    gap = float(np.mean([np.mean(a[q]) - np.mean(b[q]) for q in qs]))
    sims = np.zeros(boot)
    for q in qs:
        xa, xb = np.array(a[q]), np.array(b[q])
        sims += rng.binomial(len(xa), xa.mean(), boot) / len(xa) - rng.binomial(len(xb), xb.mean(), boot) / len(xb)
    sims /= len(qs)
    return gap, float(np.quantile(sims, 0.025)), float(np.quantile(sims, 0.975))


PAIRS = {"bad": ("good_below", "good_above"), "beers": ("beers_good_below", "beers_good_above"),
         "nothing": ("nothing_good_below", "nothing_good_above")}  # fmt: skip


def rows(runs):
    out = []
    for run in runs:
        t = json.loads((run / "meta.json").read_text())["target"]
        c = counts(run)
        head = {"run": run.name, "model": t["model"], "thinking": t["thinking"]}
        for lines, sfx in E.REF_LINES.items():
            for other, (a, b) in PAIRS.items():
                a, b = a + sfx, b + sfx
                if a in c and b in c:
                    ka, kb = kn(c[a]), kn(c[b])
                    d, lo, hi = newcombe(*ka, *kb)
                    q, qlo, qhi = qmean_gap(c[a], c[b])
                    out.append({**head, "lines": lines, "pair": other, "a": ka, "b": kb, "gap": d, "lo": lo, "hi": hi,
                                "qgap": q, "qlo": qlo, "qhi": qhi})  # fmt: skip
            if (bn := "bet_neutral" + sfx) in c:
                out.append({**head, "lines": lines, "pair": "bet_neutral", "a": kn(c[bn]), "b": None})
    return out


def main(argv):
    runs = [LOCAL / "donation_bet" / r for r in argv] or sorted((LOCAL / "donation_bet").glob("*/meta.json"))
    runs = [r.parent if r.name == "meta.json" else r for r in runs]
    print("gap = P(<= line | good below) - P(<= line | good above), pp: question mean [within-question bootstrap], pooled [Newcombe]")
    for r in rows(runs):
        head = f"{r['run']:40} {r['model']:28} think={r['thinking']!s:5} {r['lines']:6}"
        if r["b"] is None:
            print(f"{head} bet_neutral <=line {r['a'][0]}/{r['a'][1]}")
            continue
        ab = f"{r['a'][0]}/{r['a'][1]} vs {r['b'][0]}/{r['b'][1]}"
        q = f"{100 * r['qgap']:+.1f} [{100 * r['qlo']:+.0f}, {100 * r['qhi']:+.0f}]"
        print(f"{head} {r['pair']:8} {ab:22} qmean {q:18} pooled {100 * r['gap']:+.1f} [{100 * r['lo']:+.0f}, {100 * r['hi']:+.0f}]")


if __name__ == "__main__":
    main(sys.argv[1:])
