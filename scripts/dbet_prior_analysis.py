"""How much does the transcript add over the prompt-only prior? (Donation Bet investigator run)

    uv run python scripts/dbet_prior_analysis.py RUN [--x 10]

Per case: prior = mean prompt-only p_change of its (prompt, question) cell; transcript = mean p_change with the transcript;
with_prior = mean p_change when the prompt-only answer was given as an explicit prior (where run). Labels are per cell
(|Δ| > x pp). For good vs beers + good vs nothing, and each alone:
  (a) logistic regression label ~ logit(prior) + logit(transcript): transcript coefficient with a bootstrap CI clustered by
      question, in-sample log-loss gain over label ~ logit(prior), and leave-one-question-out log-loss of both models;
  (b) AUROC of the shift logit(transcript) − logit(prior);
  (c) AUROC of prior, transcript, with_prior; log-loss of each as given.
"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def logit(p):
    p = min(0.99, max(0.01, p))
    return math.log(p / (1 - p))


def auroc(y, s):
    pos, neg = [a for a, t in zip(s, y) if t], [a for a, t in zip(s, y) if not t]
    return sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg)) if pos and neg else float("nan")


def fit(X, y, l2=1e-3, iters=50):
    """logistic regression by Newton's method, X with an intercept column"""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + l2 * w
        H = X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w


def ll(X, y, w):
    p = np.clip(1 / (1 + np.exp(-X @ w)), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--x", type=int, default=10)
    a = ap.parse_args()
    run = Path(a.run)
    cs = {c["case"]: c for c in map(json.loads, (run / "cases.jsonl").open())}
    tr, wp, pr = defaultdict(list), defaultdict(list), defaultdict(list)
    for p in map(json.loads, (run / "predictions.jsonl").open()):
        if "error" not in p and "deepseek" in p["model"]:
            tr[p["case"]].append(p["p_change"])
    for p in map(json.loads, (run / "predictions_prompt_only.jsonl").open()):
        if "error" not in p:
            pr[tuple(p["cell"])].append(p["p_change"])
    if (run / "predictions_with_prior.jsonl").exists():
        for p in map(json.loads, (run / "predictions_with_prior.jsonl").open()):
            if "error" not in p:
                wp[p["case"]].append(p["p_change"])
    rows = []
    for k, c in cs.items():
        if not tr[k]:
            continue
        rows.append(
            {
                "c": c,
                "y": abs(c["delta_pp"]) > a.x,
                "prior": float(np.mean(pr[(c["orig"], c["item"])])),
                "t": float(np.mean(tr[k])),
                "wp": float(np.mean(wp[k])) if wp[k] else None,
            }
        )
    rng = random.Random(0)
    for name, keep in [
        ("beers + nothing", {"good vs beers", "good vs nothing"}),
        ("beers only", {"good vs beers"}),
        ("nothing only", {"good vs nothing"}),
        ("all three", {"good vs bad", "good vs beers", "good vs nothing"}),
    ]:
        R = [r for r in rows if r["c"]["comparison"] in keep]
        y = np.array([r["y"] for r in R], float)
        if y.min() == y.max():
            print(f"\n{name}: n={len(R)}, every label {'changes' if y[0] else 'no change'}; only log-loss is meaningful")
        X0 = np.array([[1, logit(r["prior"])] for r in R])
        X1 = np.array([[1, logit(r["prior"]), logit(r["t"])] for r in R])
        w0, w1 = fit(X0, y), fit(X1, y)
        qs = sorted({r["c"]["item"] for r in R})
        boots = []
        for _ in range(500):
            pick = [rng.choice(qs) for _ in qs]
            idx = [i for q in pick for i, r in enumerate(R) if r["c"]["item"] == q]
            if len(set(y[idx])) > 1:
                boots.append(fit(X1[idx], y[idx])[2])
        boots.sort()
        ci = (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]) if boots else (float("nan"), float("nan"))
        cv0 = cv1 = 0.0
        for q in qs:  # leave one question out
            tr_i = [i for i, r in enumerate(R) if r["c"]["item"] != q]
            te_i = [i for i, r in enumerate(R) if r["c"]["item"] == q]
            cv0 += ll(X0[te_i], y[te_i], fit(X0[tr_i], y[tr_i])) * len(te_i)
            cv1 += ll(X1[te_i], y[te_i], fit(X1[tr_i], y[tr_i])) * len(te_i)
        shift = [logit(r["t"]) - logit(r["prior"]) for r in R]
        lab = [bool(v) for v in y]

        def lls(ps, lab=lab):
            return -sum(math.log(min(0.99, max(0.01, p)) if t else 1 - min(0.99, max(0.01, p))) for t, p in zip(lab, ps)) / len(ps)

        print(f"\n{name}: n={len(R)} cases, {int(y.sum())} positive, {len(qs)} questions")
        print(
            f"  (a) label ~ logit(prior) + logit(transcript): transcript coef {w1[2]:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]"
            f" (question-clustered bootstrap), prior coef {w1[1]:+.2f}"
        )
        print(
            f"      log-loss prior-only fit {ll(X0, y, w0):.3f} -> +transcript {ll(X1, y, w1):.3f} in-sample;"
            f" leave-one-question-out {cv0 / len(R):.3f} -> {cv1 / len(R):.3f}"
        )
        print(f"  (b) AUROC of shift logit(transcript) - logit(prior): {auroc(lab, shift):.3f}")
        line = f"  (c) AUROC prior {auroc(lab, [r['prior'] for r in R]):.3f}, transcript {auroc(lab, [r['t'] for r in R]):.3f}"
        line += f" | log-loss as given: prior {lls([r['prior'] for r in R]):.3f}, transcript {lls([r['t'] for r in R]):.3f}"
        W = [r for r in R if r["wp"] is not None]
        if W:
            line += (
                f" | with explicit prior (n={len(W)}): AUROC {auroc([r['y'] for r in W], [r['wp'] for r in W]):.3f},"
                f" log-loss {lls([r['wp'] for r in W]) if len(W) == len(R) else float('nan'):.3f}"
            )
        print(line)


if __name__ == "__main__":
    main()
