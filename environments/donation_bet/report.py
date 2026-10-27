"""Leakage gaps per run folder: P(at or below the line | good cause below) - P(at or below | good cause above), for each
other side (bad / beers / nothing) and lines set, plus bet_neutral vs neutral, with 95% Newcombe CIs. Reads the latest
`measure` labels of each local run (data/envs/donation_bet/<run>/); unscorable replies are dropped.

    uv run python -m environments.donation_bet.report [RUN ...]      # default: every ref-template run
"""

import json
import math
import sys
from collections import defaultdict

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


def counts(run) -> dict[str, list[int]]:
    _, labels = latest_labels(run)
    c = defaultdict(lambda: [0, 0])
    for ep in read_episodes(run):
        v = (labels.get(ep["id"]) or {}).get("positive")
        if ep.get("error") or v is None:
            continue
        c[ep["arm"]][0] += int(v)
        c[ep["arm"]][1] += 1
    return c


def rows(runs):
    out = []
    for run in runs:
        meta = json.loads((run / "meta.json").read_text())
        t = meta["target"]
        c = counts(run)
        for lines, sfx in E.REF_LINES.items():
            for other, (a, b) in {"bad": ("good_below", "good_above"), "beers": ("beers_good_below", "beers_good_above"),
                                  "nothing": ("nothing_good_below", "nothing_good_above")}.items():  # fmt: skip
                a, b = a + sfx, b + sfx
                if c[a][1] and c[b][1]:
                    d, lo, hi = newcombe(*c[a], *c[b])
                    out.append({"run": run.name, "model": t["model"], "thinking": t["thinking"], "lines": lines, "pair": other,
                                "a": c[a], "b": c[b], "gap": d, "lo": lo, "hi": hi})  # fmt: skip
            bn = "bet_neutral" + sfx
            if c[bn][1]:
                out.append({"run": run.name, "model": t["model"], "thinking": t["thinking"], "lines": lines, "pair": "bet_neutral",
                            "a": c[bn], "b": None, "gap": c[bn][0] / c[bn][1], "lo": None, "hi": None})  # fmt: skip
    return out


def main(argv):
    runs = [LOCAL / "donation_bet" / r for r in argv] or sorted((LOCAL / "donation_bet").glob("*/meta.json"))
    runs = [r.parent if r.name == "meta.json" else r for r in runs]
    for r in rows(runs):
        if r["b"] is None:
            print(f"{r['run']:40} {r['model']:28} think={r['thinking']!s:5} {r['lines']:6} bet_neutral ≤line {r['a'][0]}/{r['a'][1]}")
        else:
            ci = f"[{100 * r['lo']:+.0f}, {100 * r['hi']:+.0f}]"
            ab = f"{r['a'][0]}/{r['a'][1]} vs {r['b'][0]}/{r['b'][1]}"
            head = f"{r['run']:40} {r['model']:28} think={r['thinking']!s:5} {r['lines']:6}"
            print(f"{head} {r['pair']:8} {ab}  gap {100 * r['gap']:+.1f} {ci}")


if __name__ == "__main__":
    main(sys.argv[1:])
