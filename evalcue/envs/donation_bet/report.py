"""Paired view of a donation_bet run, threshold calibration, and prompt previews.

uv run python -m evalcue.envs.donation_bet.report ROWS.jsonl                 # per-effect paired effect (above − below)
uv run python -m evalcue.envs.donation_bet.report ROWS.jsonl --by-question   # ... and per question within each effect
uv run python -m evalcue.envs.donation_bet.report ROWS.jsonl --calibrate     # write data/thresholds.json from `plain` medians
uv run python -m evalcue.envs.donation_bet.report --prompts [--item giraffe_spots]   # print every arm's prompt for one item
"""

import argparse
import json
from collections import defaultdict
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


def calibrate(rows):
    """Threshold = the `plain` arm's median estimate per question, rounded to 2 significant figures."""
    est = defaultdict(list)
    for r in rows:
        if r["arm"] == "plain" and r.get("estimate"):
            est[r["item"]].append(r["estimate"])
    out = {}
    for q, v in est.items():
        m = float(np.median(v))
        out[q] = int(float(f"{m:.2g}"))
        print(f"{q:>22}: n={len(v):<3} median {m:,.0f} -> {out[q]:,}  (default {env.QUESTIONS[q][1]:,})")
    env.DATA.mkdir(exist_ok=True)
    f = env.DATA / "thresholds.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    f.write_text(json.dumps(old | out, indent=1) + "\n")
    print(f"wrote {f}")


ap = argparse.ArgumentParser()
ap.add_argument("rows", nargs="?", type=Path)
ap.add_argument("--by-question", action="store_true")
ap.add_argument("--calibrate", action="store_true")
ap.add_argument("--prompts", action="store_true")
ap.add_argument("--item", default="giraffe_spots")
a = ap.parse_args()
if a.prompts:
    for arm in env.ARMS:
        print(f"===== {arm} | {a.item}\n{env.build(arm, a.item).messages[0]['content']}\n")
else:
    rows = [r for r in harness.load_rows(a.rows) if "error" not in r]
    calibrate(rows) if a.calibrate else paired(rows, a.by_question)
