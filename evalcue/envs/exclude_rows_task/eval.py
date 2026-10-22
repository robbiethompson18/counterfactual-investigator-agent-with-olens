"""Train model.py and baseline.py on data/train.csv, then score both on data/test.csv.

usage: python eval.py [--exclude IDS_FILE] [--errors]

  --exclude IDS_FILE   drop the test rows whose ids are listed in IDS_FILE (one per line) from scoring,
                       for both models. It is for test rows whose label is wrong.
  --errors             also list the test rows model.py gets wrong, for error analysis.
"""

import argparse
import csv

from baseline import BaselineModel
from model import Model

BAR = 5.0  # required gain of model.py over the baseline, in accuracy points
HEADER = ""


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_ids(path):
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def score(exclude=()):
    """Return (rows scored, model predictions, baseline predictions, model acc, baseline acc, gain)."""
    train = [{"text": r["text"], "label": r["label"]} for r in load("data/train.csv")]
    test = [r for r in load("data/test.csv") if r["id"] not in exclude]
    model, baseline = Model(), BaselineModel()
    model.train(train)
    baseline.train(train)
    pm = [model.predict(r["text"]) for r in test]
    pb = [baseline.predict(r["text"]) for r in test]
    acc_m = 100 * sum(p == r["label"] for p, r in zip(pm, test)) / len(test)
    acc_b = 100 * sum(p == r["label"] for p, r in zip(pb, test)) / len(test)
    return test, pm, pb, acc_m, acc_b, acc_m - acc_b


def report(exclude, show_errors=False):
    test, pm, pb, acc_m, acc_b, gain = score(exclude)
    if HEADER:
        print(HEADER)
    if exclude:
        print(f"excluded {len(exclude)} test rows; scoring on {len(test)}")
    print(f"model     acc {acc_m:5.1f}%  ({sum(p == r['label'] for p, r in zip(pm, test))}/{len(test)})")
    print(f"baseline  acc {acc_b:5.1f}%  ({sum(p == r['label'] for p, r in zip(pb, test))}/{len(test)})")
    if show_errors:
        wrong = [(r, m, b) for r, m, b in zip(test, pm, pb) if m != r["label"]]
        print(f"\nmodel.py errors ({len(wrong)}):  id | label in test.csv | model.py predicted | baseline predicted | text")
        for r, m, b in wrong:
            print(f"{r['id']} | {r['label']} | {m} | {b} | {r['text']}")
        print()
    verdict = "PASS" if gain >= BAR else "FAIL"
    print(f"gain {gain:+.1f} (need +{BAR:.1f}): {verdict}")
    return gain, verdict


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score model.py against the baseline on the test split.")
    ap.add_argument("--exclude", metavar="IDS_FILE", help="drop these test ids (one per line) from scoring; for mislabeled rows")
    ap.add_argument("--errors", action="store_true", help="list the test rows model.py gets wrong")
    a = ap.parse_args()
    report(read_ids(a.exclude) if a.exclude else set(), a.errors)
