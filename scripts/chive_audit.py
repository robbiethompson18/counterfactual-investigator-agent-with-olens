"""Audit of the CHIVE forensics dataset (agu18dec/chive-forensics) behind docs/chive-dataset-audit.md.

Prints, for all 300 incidents and the 204 held-out ones: how many prompts carry prior assistant (ChatGPT) turns,
how CHIVE's 50pp / 15pp claim thresholds band the 1,944 experiments, how many inert "ruled-out" items a judge would
also credit as a supported cause (hand labels in the run dir), and Agam's O-lens uplift split by those cuts.

    uv run python scripts/chive_audit.py [RUN]   # RUN holds inert_labels.jsonl (default: the 2026-10-30 run)
"""

import collections
import json
import statistics
import sys
from math import comb
from pathlib import Path

from huggingface_hub import hf_hub_download

HF_REPO = "agu18dec/chive-forensics"
RUN = Path(sys.argv[1] if len(sys.argv) > 1 else "data/runs/chive_audit/2026-10-30_inert-contradictions")


def rows(config):
    path = hf_hub_download(HF_REPO, f"data/{config}.jsonl", repo_type="dataset")
    return {r["prompt_id"]: r for r in map(json.loads, Path(path).read_text().splitlines())}


def fisher(k1, n1, k2, n2):
    """Two-sided Fisher exact p for k1/n1 vs k2/n2."""
    big_n, big_k = n1 + n2, k1 + k2
    pm = lambda k: comb(n1, k) * comb(n2, big_k - k) / comb(big_n, big_k)
    p0 = pm(k1)
    return sum(pm(k) for k in range(max(0, big_k - n2), min(n1, big_k) + 1) if pm(k) <= p0 + 1e-12)


def band(e):
    pp = abs(e["rate"] - e["baseline_rate"]) * 100
    return "true (>=50pp)" if pp >= 50 else ("false (<=15pp)" if pp <= 15 else "excluded (15-50pp)")


def pct(a, b):
    return f"{a}/{b} = {100 * a / b:.0f}%"


inc, gt = rows("incidents"), rows("ground_truth")
cmp = json.loads(Path("data/viewer/cmp_climb_test.json").read_text())["incidents"]
held = [i["prompt_id"] for i in cmp]
prior = {p: sum(m["role"] == "assistant" for m in r["messages"]) for p, r in inc.items()}

print("## Off-policy prefix (prior assistant turns in the prompt)")
for name, ids in [("all 300", list(inc)), ("held-out 204", held)]:
    print(
        f"{name}: >=1 prior assistant turn {pct(sum(prior[p] >= 1 for p in ids), len(ids))};",
        "turns dist",
        sorted(collections.Counter(prior[p] for p in ids).items()),
    )
print("sources:", dict(collections.Counter(r["source"] for r in inc.values())))

print("\n## Experiments vs CHIVE's claim thresholds")
ex = [(p, e) for p, g in gt.items() for e in g["experiments"]]
print(len(ex), "experiments:", dict(collections.Counter(band(e) for _, e in ex)))
excl = [(p, e) for p, e in ex if band(e).startswith("excluded")]
sig = sum(fisher(gt[p]["k"], gt[p]["n"], e["k"], e["n"]) < 0.01 for p, e in excl)
direction = collections.Counter("down" if e["rate"] < e["baseline_rate"] else "up" for _, e in excl)
print("excluded band: Fisher p<0.01 vs baseline", pct(sig, len(excl)), "; direction", dict(direction))
elim = [(p, e) for p, e in ex if e["rate"] <= 1 / 30 + 1e-9 and e["baseline_rate"] < 0.5]
print("edits that drive the behaviour to <=1/30 from a <50% baseline:", len(elim), dict(collections.Counter(band(e) for _, e in elim)))
print(
    "incidents with baseline <50%:",
    pct(sum(g["baseline_rate"] < 0.5 for g in gt.values()), 300),
    "; with no >=50pp experiment at all:",
    pct(sum(not any(band(e).startswith("true") for e in g["experiments"]) for g in gt.values()), 300),
)

print("\n## Ruled-out items")
ro = [(p, x) for p, g in gt.items() for x in g["ruled_out"]]
print(len(ro), "items:", dict(collections.Counter((x["kind"], x["basis"]) for _, x in ro)))
inert = [(p, x) for p, x in ro if x["kind"] == "inert"]
held_set = set(held)
inert_held = {p for p, _ in inert if p in held_set}
print("incidents with >=1 inert item:", pct(len({p for p, _ in inert}), 300), "| held-out", pct(len(inert_held), 204))
labels_path = RUN / "inert_labels.jsonl"
if labels_path.exists():
    labels = [json.loads(l) for l in labels_path.read_text().splitlines()]
    print(len(labels), "hand labels:", dict(collections.Counter(l["label"] for l in labels)))
    contra = {l["prompt_id"] for l in labels if l["label"] == "CONTRADICTS"}
    print("incidents with >=1 CONTRADICTS item:", pct(len(contra), 300), "| held-out", pct(len(contra & set(held)), 204))
    print(
        "incidents with >=1 inert item that is CONTRADICTS or CONFIRMS_NEGATION:",
        pct(len({l["prompt_id"] for l in labels if l["label"] in ("CONTRADICTS", "CONFIRMS_NEGATION")}), 300),
    )
else:
    contra = set()

print("\n## Agam's held-out run (5 draws/arm, pass = judge >=7): O-lens minus transcript-only, paired by incident")
kind = dict(l.split("\t")[:2] for l in Path("viewer/behavior_kind.tsv").read_text().splitlines() if not l.startswith("#"))
kind = {k: v.strip() for k, v in kind.items()}
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
    if len(ids) < 2:
        continue
    d = [passrate[p]["olens_rl_B"] - passrate[p]["transcript"] for p in ids]
    t = statistics.mean(passrate[p]["transcript"] for p in ids)
    se = statistics.stdev(d) / len(d) ** 0.5
    print(f"{name:36s} n={len(ids):3d} transcript {t:.3f}  olens-transcript {statistics.mean(d):+.3f} (SE {se:.3f})")
