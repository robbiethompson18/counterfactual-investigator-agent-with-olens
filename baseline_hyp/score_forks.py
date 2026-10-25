"""Score counterfactual-prediction forks against observed rate changes.

Label for an edit on a variant: delta = arm rate - baseline rate (percentage points, same variant).
  up      delta >= +T      down   delta <= -T      neither otherwise
Also scored against the pooled label (arm vs baseline pooled over all variants) with --pooled.

Per format: Brier (and a constant base-rate predictor's Brier, "brier_ref"), log loss, AUC,
expected calibration error (10 bins), accuracy; bal_* report accuracy and yes-rate; point reports MAE
and correlation of predicted vs observed change.

python score_forks.py forks/v41 [--min-n 8] [--pooled] [--single] [--json]
--single keeps only edits that change one thing (cf.single_change). brier_bal / brier_3class_bal weight
classes equally (constant 0.5 forecast = 0.25; uniform 1/3 forecast = 0.667).
"""

import collections
import glob
import json
import math
import sys
from pathlib import Path


def pooled_delta():
    cf = json.loads(Path("cf_index.json").read_text())
    idx = json.loads(Path("index.json").read_text())
    base = {e["env"]: (e["arm"], e["k"] / e["n"]) for e in idx}
    return {(env, arm): 100 * (k / n - base[env][1]) for env, d in cf.items() for arm, (k, n) in d["arms"].items()}


def auc(p, y):
    pos = [a for a, b in zip(p, y) if b]
    neg = [a for a, b in zip(p, y) if not b]
    if not pos or not neg:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg))


def ece(p, y, bins=10):
    tot, e = len(p), 0.0
    rel = []
    for i in range(bins):
        sel = [(a, b) for a, b in zip(p, y) if (i / bins <= a < (i + 1) / bins) or (i == bins - 1 and a == 1)]
        if sel:
            mp = sum(a for a, _ in sel) / len(sel)
            my = sum(b for _, b in sel) / len(sel)
            e += len(sel) / tot * abs(mp - my)
            rel.append([round(mp, 3), round(my, 3), len(sel)])
    return e, rel


def balanced_brier(p, y):
    """Brier with each class weighted equally (so a mostly-decrease label set can't be won by always saying decrease).
    A constant 0.5 forecast scores 0.25."""
    pos = [(a - 1) ** 2 for a, b in zip(p, y) if b]
    neg = [a**2 for a, b in zip(p, y) if not b]
    if not pos or not neg:
        return None
    return round(0.5 * sum(pos) / len(pos) + 0.5 * sum(neg) / len(neg), 4)


def binary(p, y):
    n = len(p)
    base = sum(y) / n
    brier = sum((a - b) ** 2 for a, b in zip(p, y)) / n
    ll = -sum(b * math.log(max(a, 1e-3)) + (1 - b) * math.log(max(1 - a, 1e-3)) for a, b in zip(p, y)) / n
    e, rel = ece(p, y)
    return {
        "n": n,
        "pos_rate": round(base, 3),
        "brier": round(brier, 4),
        "brier_ref": round(base * (1 - base), 4),
        "brier_bal": balanced_brier(p, y),
        "logloss": round(ll, 4),
        "auc": None if auc(p, y) is None else round(auc(p, y), 3),
        "ece": round(e, 3),
        "acc": round(sum((a >= 0.5) == b for a, b in zip(p, y)) / n, 3),
        "reliability": rel,
    }


def load(d):
    return [json.loads(Path(f).read_text()) for f in glob.glob(f"{d}/*/*.json")]


SINGLE = None


def single_set():
    global SINGLE
    if SINGLE is None:
        cf = json.loads(Path("cf_index.json").read_text())
        SINGLE = {(v, c["arm"]) for d in cf.values() for v, cs in d["views"].items() for c in cs if c.get("single")}
    return SINGLE


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    c = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    d = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return round(c / d, 3) if d else None


def within(rs):
    byinv = collections.defaultdict(list)
    for r in rs:
        byinv[r["inv"]].append(r)
    ws = [spearman([r["parsed"]["rate"] for r in v], [r["delta"] for r in v]) for v in byinv.values() if len(v) >= 5]
    ws = [w for w in ws if w is not None]
    return round(sum(ws) / len(ws), 3) if ws else None


def score(rows, min_n=0, pooled=False, single=False):
    pd = pooled_delta() if pooled else None
    ss = single_set() if single else None
    by = collections.defaultdict(list)
    for r in rows:
        if r["parsed"] is None or r["arm_n"] < min_n or (ss is not None and (r["view"], r["arm"]) not in ss):
            continue
        r = dict(r)
        delta = pd[(r["env"], r["arm"])] if pooled else r["delta_pp"]
        T = r["thresh"]
        r["lab"] = "up" if delta >= T else "down" if delta <= -T else "neither"
        r["delta"] = delta
        by[r["fmt"]].append(r)
    out = {}
    for fmt, rs in by.items():
        if fmt == "probs":
            o = {}
            for c, k in (("up", "p_increase"), ("down", "p_decrease"), ("neither", "p_neither")):
                o[c] = binary([r["parsed"][k] for r in rs], [r["lab"] == c for r in rs])
            o["brier_3class"] = round(
                sum(
                    sum(
                        (r["parsed"][k] - (r["lab"] == c)) ** 2
                        for c, k in (("up", "p_increase"), ("down", "p_decrease"), ("neither", "p_neither"))
                    )
                    for r in rs
                )
                / len(rs),
                4,
            )
            K = (("up", "p_increase"), ("down", "p_decrease"), ("neither", "p_neither"))
            cls = {c: [r for r in rs if r["lab"] == c] for c, _ in K}
            o["brier_3class_bal"] = round(
                sum(sum(sum((r["parsed"][k] - (r["lab"] == c2)) ** 2 for c2, k in K) for r in cls[c]) / len(cls[c]) for c, _ in K if cls[c])
                / sum(1 for c, _ in K if cls[c]),
                4,
            )
            o["brier_3class_ref"] = round(sum(o[c]["brier_ref"] for c in ("up", "down", "neither")), 4)
            o["mean_pred"] = {
                c: round(sum(r["parsed"][k] for r in rs) / len(rs), 3)
                for c, k in (("up", "p_increase"), ("down", "p_decrease"), ("neither", "p_neither"))
            }
            aucs = [o[c]["auc"] for c, _ in K if o[c]["auc"] is not None]
            o["macro_auc"] = round(sum(aucs) / len(aucs), 3) if aucs else None
            # signed expected direction vs observed change: threshold-free, prior-insensitive
            o["spearman_dir"] = spearman([r["parsed"]["p_increase"] - r["parsed"]["p_decrease"] for r in rs], [r["delta"] for r in rs])
            # within one investigation (same baseline), rank edits by expected direction vs observed change; averaged.
            # Removes between-variant base-rate differences as well as class skew.
            byinv = collections.defaultdict(list)
            for r in rs:
                byinv[r["inv"]].append(r)
            ws = [
                spearman([r["parsed"]["p_increase"] - r["parsed"]["p_decrease"] for r in v], [r["delta"] for r in v])
                for v in byinv.values()
                if len(v) >= 5
            ]
            ws = [w for w in ws if w is not None]
            o["spearman_within"] = round(sum(ws) / len(ws), 3) if ws else None
            o["acc_argmax"] = round(
                sum(
                    max((("up", "p_increase"), ("down", "p_decrease"), ("neither", "p_neither")), key=lambda ck: r["parsed"][ck[1]])[0]
                    == r["lab"]
                    for r in rs
                )
                / len(rs),
                3,
            )
            out[fmt] = o
        elif fmt in ("q_up", "q_down"):
            c = "up" if fmt == "q_up" else "down"
            out[fmt] = binary([r["parsed"]["p"] for r in rs], [r["lab"] == c for r in rs])
        elif fmt in ("bal_up", "bal_down"):
            truth = [(r["delta"] > 0) if fmt == "bal_up" else (r["delta"] < 0) for r in rs]
            yes = [r["parsed"]["answer"] == "yes" for r in rs]
            tp = [a for a, b in zip(yes, truth) if b]
            tn = [not a for a, b in zip(yes, truth) if not b]
            bal = round(0.5 * (sum(tp) / len(tp) + sum(tn) / len(tn)), 3) if tp and tn else None
            out[fmt] = {
                "n": len(rs),
                "acc": round(sum(a == b for a, b in zip(yes, truth)) / len(rs), 3),
                "bal_acc": bal,
                "yes_rate": round(sum(yes) / len(rs), 3),
                "true_rate": round(sum(truth) / len(rs), 3),
            }
        elif fmt == "point":
            pred = [r["parsed"]["rate"] - 100 * r["base_k"] / r["base_n"] for r in rs]
            obs = [r["delta"] for r in rs]
            mp, mo = sum(pred) / len(pred), sum(obs) / len(obs)
            cov = sum((a - mp) * (b - mo) for a, b in zip(pred, obs))
            sd = math.sqrt(sum((a - mp) ** 2 for a in pred) * sum((b - mo) ** 2 for b in obs)) or 1
            out[fmt] = {
                "n": len(rs),
                "mae_pp": round(sum(abs(r["parsed"]["rate"] - 100 * r["arm_k"] / r["arm_n"]) for r in rs) / len(rs), 1),
                "corr_delta": round(cov / sd, 3),
                "spearman": spearman(pred, obs),
                "spearman_within": within(rs),
            }
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    MIN_N = int(sys.argv[sys.argv.index("--min-n") + 1]) if "--min-n" in sys.argv else 0
    if MIN_N:
        args = [a for a in args if a != str(MIN_N)]
    POOLED = "--pooled" in sys.argv
    for d in args:
        s = score(load(d), MIN_N, POOLED, "--single" in sys.argv)
        if "--json" in sys.argv:
            print(json.dumps({d: s}, indent=1))
            continue
        print(f"== {d}" + (" (pooled labels)" if POOLED else "") + (f" (arm n>={MIN_N})" if MIN_N else ""))
        for fmt, o in s.items():
            if fmt == "probs":
                print(
                    f"  probs: brier3={o['brier_3class']} acc_argmax={o['acc_argmax']}  "
                    + "  ".join(
                        f"{c}: brier {o[c]['brier']} (ref {o[c]['brier_ref']}) auc {o[c]['auc']} ece {o[c]['ece']}"
                        for c in ("up", "down", "neither")
                    )
                )
            elif fmt in ("q_up", "q_down"):
                print(
                    f"  {fmt}: n={o['n']} pos={o['pos_rate']} brier {o['brier']} (ref {o['brier_ref']})"
                    f" auc {o['auc']} ece {o['ece']} acc {o['acc']}"
                )
            else:
                print(f"  {fmt}: " + " ".join(f"{k}={v}" for k, v in o.items()))
