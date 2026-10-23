"""Paired view of a donation_bet run, threshold calibration, and prompt previews.

uv run python -m evalcue.envs.donation_bet.report ROWS.jsonl                 # per-effect paired effect (above − below)
uv run python -m evalcue.envs.donation_bet.report ROWS.jsonl --by-question   # ... and per question within each effect
uv run python -m evalcue.envs.donation_bet.report ROWS.jsonl --calibrate     # write thresholds.json from `plain` medians
uv run python -m evalcue.envs.donation_bet.report --prompts [--item giraffe_spots]   # print every arm's prompt for one item
"""

import argparse
import json
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import numpy as np

from evalcue import harness
from evalcue.envs import donation_bet as env


def rate(rows):
    sc = [r for r in rows if r.get("positive") is not None]
    return sum(r["positive"] for r in sc), len(sc)


def paired(rows, by_question: bool):
    """For each effect: P(over | above pays) − P(over | below pays), plus the median log10(estimate/threshold) shift."""
    cells = defaultdict(list)
    for r in rows:
        cells[r["arm"]].append(r)
    for ctrl in ("plain", "anchor"):
        if cells[ctrl]:
            k, n = rate(cells[ctrl])
            print(f"{ctrl:>22}: over {k}/{n} = {k / max(n, 1):.0%}")
    print(f"{'effect':>22}  {'tier':>8}  {'over|above':>12}  {'over|below':>12}  {'Δ':>6}  {'Δ median log10':>14}")
    for e in env.EFFECTS:
        up, dn = cells[f"above+{e}"], cells[f"below+{e}"]
        if not up and not dn:
            continue
        (ku, nu), (kd, nd) = rate(up), rate(dn)
        lu = [r["log_ratio"] for r in up if r.get("log_ratio") is not None]
        ld = [r["log_ratio"] for r in dn if r.get("log_ratio") is not None]
        dlog = np.median(lu) - np.median(ld) if lu and ld else float("nan")
        tier = (up or dn)[0]["meta"]["tier"]
        pu, pd = ku / max(nu, 1), kd / max(nd, 1)
        print(f"{e:>22}  {tier:>8}  {ku:>4}/{nu:<3}{pu:>4.0%}  {kd:>4}/{nd:<3}{pd:>4.0%}  {pu - pd:>+6.0%}  {dlog:>+14.2f}")
        if by_question:
            for q in env.QUESTIONS:
                qu = [r for r in up if r["item"] == q]
                qd = [r for r in dn if r["item"] == q]
                if qu or qd:
                    (a, b), (c, f) = rate(qu), rate(qd)
                    print(f"{'':>24}{q:<22} above {a}/{b}  below {c}/{f}")


def split_point(v: list[float]) -> int:
    """The roundest number strictly between two of the estimates that puts the split closest to 50/50. Never equal to an
    estimate: Qwen piles onto round numbers (8/10 say exactly 20 billion hot dogs), and a line on the pile is a tie."""
    xs = sorted(set(v))
    best = None
    for lo, hi in pairwise(xs):
        mid = float(np.sqrt(lo * hi)) if lo > 0 else (lo + hi) / 2
        for sig in range(1, 6):  # roundest representation that still sits strictly inside the gap
            c = float(f"{mid:.{sig}g}")
            if lo < c < hi:
                break
        frac = np.mean([x > c for x in v])
        key = (abs(frac - 0.5), sig)
        if best is None or key < best[0]:
            best = (key, int(c))
    assert best is not None, "need at least two distinct estimates"
    return best[1]


def calibrate(rows):
    """Threshold per question from the `plain` arm's estimates (see split_point)."""
    est = defaultdict(list)
    for r in rows:
        if r["arm"] == "plain" and r.get("estimate"):
            est[r["item"]].append(r["estimate"])
    out = {}
    for q, v in est.items():
        out[q] = split_point(v)
        over = np.mean([x > out[q] for x in v])
        print(f"{q:>22}: n={len(v):<3} median {np.median(v):,.0f} -> {out[q]:,} ({over:.0%} over)  (default {env.QUESTIONS[q][1]:,})")
    env.DATA.mkdir(exist_ok=True)
    f = env.DATA / "thresholds.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    f.write_text(json.dumps(old | out, indent=1) + "\n")
    print(f"wrote {f}")


def median_split(up, down) -> tuple[tuple[float, int], tuple[float, int]]:
    """Over-the-pair-median counts for the two arms of a pair. The line anchors both arms alike, so splitting at the pair's own
    pooled median measures only the direction effect. Estimates exactly on the median (Qwen piles onto round numbers) count
    half, which makes the gap equal to 2·AUC − 1 up to the split (AUC = P(above-pays estimate > below-pays estimate))."""
    u = [r["estimate"] for r in up if r.get("estimate")]
    d = [r["estimate"] for r in down if r.get("estimate")]
    if not u or not d:
        return (0, len(u)), (0, len(d))
    m = float(np.median(u + d))
    k = lambda v: sum((x > m) + 0.5 * (x == m) for x in v)
    return (k(u), len(u)), (k(d), len(d))


def grid(rows) -> dict:
    """Per (question, effect) cell: over-the-pair-median counts in each direction (see median_split), plus over-the-line
    counts, for a heatmap. Also the anchor/plain over-the-line rates."""
    cells = defaultdict(list)
    for r in rows:
        cells[r["arm"], r["item"]].append(r)
    out = {"questions": list(env.QUESTIONS), "effects": env.EFFECTS, "thresholds": env.thresholds(), "cells": {}, "controls": {}}
    out["tiers"] = {e: env.CAUSES[e][1] for e in env.EFFECTS}
    for q in env.QUESTIONS:
        for e in env.EFFECTS:
            up, down = cells[f"above+{e}", q], cells[f"below+{e}", q]
            med_up, med_down = median_split(up, down)
            out["cells"][f"{q}|{e}"] = {"above": med_up, "below": med_down, "line_above": rate(up), "line_below": rate(down)}
        out["controls"][q] = {c: rate(cells[c, q]) for c in ("anchor", "plain")}
    return out


ap = argparse.ArgumentParser()
ap.add_argument("rows", nargs="?", type=Path)
ap.add_argument("--by-question", action="store_true")
ap.add_argument("--calibrate", action="store_true")
ap.add_argument("--prompts", action="store_true")
ap.add_argument("--grid-json", type=Path, help="write per (question, effect) counts for a heatmap")
ap.add_argument("--item", default="giraffe_spots")
a = ap.parse_args()
if a.prompts:
    for arm in env.ARMS:
        print(f"===== {arm} | {a.item}\n{env.build(arm, a.item).messages[0]['content']}\n")
else:
    rows = [r for r in harness.load_rows(a.rows) if "error" not in r]
    if a.grid_json:
        a.grid_json.write_text(json.dumps(grid(rows)))
    elif a.calibrate:
        calibrate(rows)
    else:
        paired(rows, a.by_question)
