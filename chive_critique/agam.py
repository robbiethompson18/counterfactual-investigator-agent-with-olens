"""The same critique on Agam's 300-incident Qwen3.6-27B CHIVE run (agu18dec/chive-forensics), the data we started from.
Adam's release is the primary evidence (adam.py); this is the appendix, plus the only full contradiction count so far."""

import collections
import statistics
from math import comb
from pathlib import Path

from chive_critique.data import agam, band, jsonl, pct, run_file

BEHAVIOR_KIND = Path("viewer/behavior_kind.tsv")  # our single-rater labels of the 204 held-out behaviours: wrong / content / policy


def fisher(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact p for k1/n1 vs k2/n2."""
    big_n, big_k = n1 + n2, k1 + k2
    pm = lambda k: comb(n1, k) * comb(n2, big_k - k) / comb(big_n, big_k)
    p0 = pm(k1)
    return sum(pm(k) for k in range(max(0, big_k - n2), min(n1, big_k) + 1) if pm(k) <= p0 + 1e-12)


def report():
    inc, gt, cmp = agam()
    held = [i["prompt_id"] for i in cmp]
    prior = {p: sum(m["role"] == "assistant" for m in r["messages"]) for p, r in inc.items()}

    print("## Off-policy prefix (prior assistant turns in the prompt)")
    for name, ids in [("all 300", list(inc)), ("held-out 204", held)]:
        turns = sorted(collections.Counter(prior[p] for p in ids).items())
        print(f"  {name}: >=1 prior assistant turn {pct(sum(prior[p] >= 1 for p in ids), len(ids))}; turns {turns}")

    print("## Experiments vs the 50pp / 15pp rule")
    ex = [(p, e["baseline_rate"], e["rate"], e["k"], e["n"]) for p, g in gt.items() for e in g["experiments"]]
    print(f"  {len(ex)} experiments:", dict(collections.Counter(band(b, r) for _, b, r, _, _ in ex)))
    excl = [(p, b, r, k, n) for p, b, r, k, n in ex if band(b, r) == "excluded"]
    sig = sum(fisher(gt[p]["k"], gt[p]["n"], k, n) < 0.01 for p, b, r, k, n in excl)
    direction = dict(collections.Counter("down" if r < b else "up" for _, b, r, _, _ in excl))
    print(f"  excluded band: Fisher p<0.01 vs baseline {pct(sig, len(excl))}; direction {direction}")
    stuck = {p for p, b, r, k, n in ex if k == 0 and 0.10 < b < 0.50}
    n_stuck = sum(1 for p, b, r, k, n in ex if k == 0 and 0.10 < b < 0.50)
    has_true = {p for p, b, r, _, _ in ex if band(b, r) == "true"}
    print(f"  baseline 10-50% -> 0/n: {pct(n_stuck, len(ex))} of experiments, in {pct(len(stuck), 300)} of incidents;", end=" ")
    print(f"{pct(len(stuck - has_true), 300)} have one and no >=50pp experiment")

    print("## Ruled-out items and the hand labels (agam_inert_labels.jsonl)")
    ro = [(p, x) for p, g in gt.items() for x in g["ruled_out"]]
    print(f"  {len(ro)} items:", dict(collections.Counter((x["kind"], x["basis"]) for _, x in ro)))
    labels = jsonl(run_file("agam_inert_labels.jsonl"))
    print(f"  {len(labels)} inert items labelled:", dict(collections.Counter(item["label"] for item in labels)))
    contra = {item["prompt_id"] for item in labels if item["label"] == "CONTRADICTS"}
    neg = {item["prompt_id"] for item in labels if item["label"] == "CONFIRMS_NEGATION"}
    print(f"  incidents with a CONTRADICTS item: {pct(len(contra), 300)} | held-out {pct(len(contra & set(held)), 204)}")
    print(f"  incidents with a negation-phrased cause: {pct(len(neg), 300)}; either: {pct(len(contra | neg), 300)}")

    print("## Agam's held-out run (5 draws/arm, pass = judge >= 7): O-lens minus transcript-only, paired by incident")
    kind = {k: v.strip() for k, v in (line.split("\t")[:2] for line in BEHAVIOR_KIND.read_text().splitlines() if not line.startswith("#"))}
    passrate = {i["prompt_id"]: {a["arm"]: statistics.mean(s >= 7 for _, s in a["draws"]) for a in i["arms"]} for i in cmp}
    cuts = [
        ("off-policy (>=1 prior asst turn)", lambda p: prior[p] >= 1),
        ("on-policy (0 prior asst turns)", lambda p: prior[p] == 0),
        ("behaviour = wrong", lambda p: kind.get(p) == "wrong"),
        ("behaviour = content", lambda p: kind.get(p) == "content"),
        ("behaviour = policy", lambda p: kind.get(p) == "policy"),
        ("has CONTRADICTS item", lambda p: p in contra),
        ("no CONTRADICTS item", lambda p: p not in contra),
        ("all", lambda p: True),
    ]
    for name, f in cuts:
        ids = [p for p in held if f(p)]
        d = [passrate[p]["olens_rl_B"] - passrate[p]["transcript"] for p in ids]
        t = statistics.mean(passrate[p]["transcript"] for p in ids)
        se = 100 * statistics.stdev(d) / len(d) ** 0.5
        print(f"  {name:36s} n={len(ids):3d} transcript {t:.3f}  olens-transcript {100 * statistics.mean(d):+.1f}pp (SE {se:.1f})")
    print("  hand labels of the 204 held-out behaviours:", dict(collections.Counter(kind.values())))
