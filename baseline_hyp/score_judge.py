""" "Hypothesis generation" scores from judge/<set>: did investigations name the feature each edit changes?
An edit is "effective" if |arm rate - baseline rate| >= T pp on that variant (default 20).
recall = effective edits named; false_alarm = ineffective edits named; auc = P(named | effective) vs
P(named | ineffective) as a ranking of named-fraction per (view, edit) against |delta|.

python score_judge.py judge/v41 [--min-n 8]
"""

import collections
import glob
import json
import sys
from pathlib import Path

from score_forks import auc


def load(d):
    return [json.loads(Path(f).read_text()) for f in glob.glob(f"{d}/*/*.json")]


def score(rows, T=20, min_n=0):
    rows = [r for r in rows if r["parsed"] is not None and r["arm_n"] >= min_n]
    out = {}
    for env in ["all"] + sorted({r["env"] for r in rows}):
        rs = [r for r in rows if env == "all" or r["env"] == env]
        eff = [r for r in rs if abs(r["delta_pp"]) >= T]
        ineff = [r for r in rs if abs(r["delta_pp"]) < T]
        per = collections.defaultdict(list)
        for r in rs:
            per[(r["view"], r["arm"])].append(r)
        frac = [sum(x["parsed"]["named"] for x in v) / len(v) for v in per.values()]
        lab = [abs(v[0]["delta_pp"]) >= T for v in per.values()]
        a = auc(frac, lab)
        out[env] = {
            "n": len(rs),
            "named": round(sum(r["parsed"]["named"] for r in rs) / max(1, len(rs)), 3),
            "recall": round(sum(r["parsed"]["named"] for r in eff) / len(eff), 3) if eff else None,
            "n_eff": len(eff),
            "false_alarm": round(sum(r["parsed"]["named"] for r in ineff) / len(ineff), 3) if ineff else None,
            "n_ineff": len(ineff),
            "auc": None if a is None else round(a, 3),
        }
    return out


def per_edit(rows):
    out = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r["parsed"] is not None:
            out[f"{r['view']}|{r['arm']}"][0] += r["parsed"]["named"]
            out[f"{r['view']}|{r['arm']}"][1] += 1
    return out


if __name__ == "__main__":
    mn = int(sys.argv[sys.argv.index("--min-n") + 1]) if "--min-n" in sys.argv else 0
    for d in [a for a in sys.argv[1:] if a.startswith("judge")]:
        print("==", d)
        for env, o in score(load(d), min_n=mn).items():
            print(f"  {env:32s} " + " ".join(f"{k}={v}" for k, v in o.items()))
