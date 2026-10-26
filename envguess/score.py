# ruff: noqa: E501
"""Score judge verdicts: per arm, how often investigators name the real edit vs the null edits.

Per investigation and edit: named (any hypothesis or proposed test names the feature the edit changes) and rank (the
first hypothesis number that names it). Pairwise score per (real, null) pair: 1 if the real edit is named at a better
rank than the null, 0.5 on a tie (same rank or neither named), 0 otherwise; 0.5 = can't tell them apart.
Means are over settings (each setting weighs the same), 95% CIs by bootstrapping settings.

  ~/workspace/donehack/.venv/bin/python envguess/score.py --judge data/envguess/judge/v1 --out data/envguess/summary_v1.json
"""

import argparse
import collections
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SETTINGS = REPO / "data" / "envguess" / "settings"
ARMS = ["env", "env_tx", "tx"]
INF = 99


def load(judge_dir):
    by_inv = collections.defaultdict(dict)  # (arm, view, seed) -> label -> (named, rank)
    for f in Path(judge_dir).glob("*.json"):
        d = json.loads(f.read_text())
        v = d["verdict"]
        rank = min(v["which"]) if v["named"] and v["which"] else (1 if v["named"] else INF)
        by_inv[(d["arm"], d["view"], d["seed"])][d["label"]] = (v["named"], rank)
    return by_inv


def per_inv(by_inv):
    rows = []
    for (arm, view, seed), labs in by_inv.items():
        sid, cid = view.rsplit("__", 1)
        s = json.loads((SETTINGS / sid / "setting.json").read_text())
        if set(labs) != set(s["edits"]):
            continue
        real = [labs[lab] for lab, e in s["edits"].items() if e["role"] == "real"]
        null = [labs[lab] for lab, e in s["edits"].items() if e["role"] == "null"]
        pair = [1.0 if r[1] < n[1] else 0.5 if r[1] == n[1] else 0.0 for r in real for n in null]
        rows.append(
            {
                "arm": arm,
                "setting": sid,
                "family": s["family"],
                "case": cid,
                "seed": seed,
                "real_named": sum(r[0] for r in real) / len(real),
                "null_named": sum(n[0] for n in null) / len(null),
                "real_top1": sum(r[1] == 1 for r in real) / len(real),
                "null_top1": sum(n[1] == 1 for n in null) / len(null),
                "pairwise": sum(pair) / len(pair),
            }
        )
    return rows


METRICS = ["real_named", "null_named", "real_top1", "null_top1", "pairwise"]


def setting_means(rows):
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["arm"], r["setting"])].append(r)
    return {k: {m: sum(r[m] for r in v) / len(v) for m in METRICS} | {"n": len(v), "family": v[0]["family"]} for k, v in g.items()}


def pooled(sm, settings, arm, rng, boots=2000):
    out = {}
    for m in METRICS:
        xs = [sm[(arm, s)][m] for s in settings if (arm, s) in sm]
        mean = sum(xs) / len(xs)
        bs = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(boots))
        out[m] = [mean, bs[int(0.025 * boots)], bs[int(0.975 * boots)]]
    return out


def paired_diff(sm, settings, a, b, m, rng, boots=2000):
    ds = [sm[(a, s)][m] - sm[(b, s)][m] for s in settings if (a, s) in sm and (b, s) in sm]
    mean = sum(ds) / len(ds)
    bs = sorted(sum(rng.choice(ds) for _ in ds) / len(ds) for _ in range(boots))
    return [mean, bs[int(0.025 * boots)], bs[int(0.975 * boots)]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-family", default="", help="comma-separated families to drop, e.g. realism")
    a = ap.parse_args()
    drop = set(filter(None, a.exclude_family.split(",")))
    rows = [r for r in per_inv(load(a.judge)) if r["family"] not in drop]
    sm = setting_means(rows)
    settings = sorted({s for _, s in sm})
    fams = sorted({v["family"] for v in sm.values()})
    rng = random.Random(0)
    res = {
        "n_investigations": len(rows),
        "settings": settings,
        "pooled": {arm: pooled(sm, settings, arm, rng) for arm in ARMS},
        "by_family": {f: {arm: pooled(sm, [s for s in settings if sm[(arm, s)]["family"] == f], arm, rng) for arm in ARMS} for f in fams},
        "diffs": {
            f"{x}-{y}": {m: paired_diff(sm, settings, x, y, m, rng) for m in METRICS}
            for x, y in [("env_tx", "env"), ("tx", "env"), ("env_tx", "tx")]
        },
        "per_setting": {f"{arm}|{s}": v for (arm, s), v in sm.items()},
        "rows": rows,
    }
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"{len(rows)} investigations, {len(settings)} settings")
    for arm in ARMS:
        p = res["pooled"][arm]
        print(f"{arm:7s} " + "  ".join(f"{m} {p[m][0]:.2f} [{p[m][1]:.2f},{p[m][2]:.2f}]" for m in METRICS))
    for k, v in res["diffs"].items():
        print(f"{k:12s} " + "  ".join(f"{m} {v[m][0]:+.2f} [{v[m][1]:+.2f},{v[m][2]:+.2f}]" for m in METRICS))
    for f in fams:
        print(
            f,
            "  ".join(
                f"{arm}: real {res['by_family'][f][arm]['real_named'][0]:.2f} null {res['by_family'][f][arm]['null_named'][0]:.2f} pair {res['by_family'][f][arm]['pairwise'][0]:.2f}"
                for arm in ARMS
            ),
        )
