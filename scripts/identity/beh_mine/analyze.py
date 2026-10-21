"""Per-probe answer distributions per condition, shift vs `none`, Fisher p, CoT-mention rates.

uv run --with scipy python scripts/identity/beh_mine/analyze.py <tag[,tag]> [stage-name-for-progress.tsv]
Prints a ranked table; writes summary_<tags>.json; appends one line per probe to progress.tsv if a stage name is given.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import fisher_exact

sys.path.insert(0, "scripts/identity/beh_mine")
from probes import PROBES, score

OUT = Path("data/runs/identity/2026-10-21_beh_mine")
tags = sys.argv[1].split(",")
stage = sys.argv[2] if len(sys.argv) > 2 else None
rows = [json.loads(line) for t in tags for line in (OUT / f"raw_{t}.jsonl").open()]
HAND = OUT / "hand_cot.jsonl"  # CoTs read by me / Claude subagents: {"probe","cond","i","mentions": bool}
hand = defaultdict(list)
if HAND.exists():
    for line in HAND.open():
        h = json.loads(line)
        if h["tag"] in tags:
            hand[h["probe"], h["cond"]].append(h)

by = defaultdict(list)
for r in rows:  # re-score from the raw text so scorer fixes apply to old runs
    r["label"] = score(r["probe"], r["content"], r["reasoning"])
    by[r["probe"], r["cond"]].append(r)


def dist(rs):
    return dict(Counter(r["label"] for r in rs).most_common())


def tvd(a, b):
    na, nb = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a.get(k, 0) / na - b.get(k, 0) / nb) for k in set(a) | set(b))


def shift(rs, base):
    """Fisher p for the label most over-represented in rs relative to base."""
    a, b = dist(rs), dist(base)
    lab = max(a, key=lambda k: a[k] / len(rs) - b.get(k, 0) / len(base))
    k, kb = a[lab], b.get(lab, 0)
    return lab, k / len(rs), kb / len(base), float(fisher_exact([[k, len(rs) - k], [kb, len(base) - kb]])[1])


def top(d, n):
    return " / ".join(f"{k} {v}" for k, v in list(d.items())[:3]) + f" (n={n})"


summary = []
for p in PROBES:
    if (p, "none") not in by:
        continue
    base = by[p, "none"]
    e = {"probe": p, "prompt": PROBES[p][0], "scorer": PROBES[p][1], "cells": {}}
    for c in ("none", "qwen", "claude", "gemini", "helpful", "nova"):
        rs = by[p, c]
        if not rs:
            continue
        cell = {
            "n": len(rs),
            "dist": dist(rs),
            "answer_names_vendor": sum(bool(r["answer_vendors"]) for r in rs) / len(rs),
            "cot_names_cued": sum(r["cot_names_cued"] for r in rs) / len(rs),
            "truncated": sum(r["finish_reason"] != "stop" for r in rs),
        }
        if hand[p, c]:
            cell["cot_self_id_hand"] = sum(h["mentions"] for h in hand[p, c]) / len(hand[p, c])
            cell["hand_n"] = len(hand[p, c])
        if c != "none":
            lab, rate, brate, pv = shift(rs, base)
            cell |= {"tvd_vs_none": tvd(dist(rs), dist(base)), "shift_label": lab, "shift_rate": rate, "base_rate": brate, "fisher_p": pv}
        e["cells"][c] = cell
    cl, ge = e["cells"]["claude"], e["cells"]["gemini"]
    e["effect"] = max(cl["tvd_vs_none"], ge["tvd_vs_none"])
    e["min_p"] = min(cl["fisher_p"], ge["fisher_p"])
    # CoT mention averaged over the shifted identity cells (weighted by how much each shifted)
    w = [(cl["tvd_vs_none"], cl), (ge["tvd_vs_none"], ge)]
    tot = sum(x for x, _ in w) or 1
    key = "cot_self_id_hand" if all("cot_self_id_hand" in c for _, c in w) else "cot_names_cued"
    e["cot_mention"] = sum(x * c[key] for x, c in w) / tot
    e["cot_mention_source"] = key
    e["cot_names_cued"] = sum(x * c["cot_names_cued"] for x, c in w) / tot
    e["answer_names_vendor"] = sum(x * c["answer_names_vendor"] for x, c in w) / tot
    e["shift"] = max(cl["shift_rate"] - cl["base_rate"], ge["shift_rate"] - ge["base_rate"])
    e["silent_score"] = e["shift"] * (1 - e["cot_mention"])
    summary.append(e)

summary.sort(key=lambda e: (e["min_p"], -e["effect"]))
name = "_".join(tags)
(OUT / f"summary_{name}.json").write_text(json.dumps(summary, indent=1))
print(f"{'probe':22} {'eff':>5} {'p':>8} {'cotM':>5} {'cotV':>5} {'ansV':>5}  none | claude | gemini")
for e in summary:
    c = e["cells"]
    print(
        f"{e['probe']:22} {e['effect']:5.2f} {e['min_p']:8.1e} {e['cot_mention']:5.2f} {e['cot_names_cued']:5.2f} {e['answer_names_vendor']:5.2f}  "
        f"{top(c['none']['dist'], c['none']['n'])} | {top(c['claude']['dist'], c['claude']['n'])} | {top(c['gemini']['dist'], c['gemini']['n'])}"
    )
if stage:
    tsv = OUT / "progress.tsv"
    new = not tsv.exists()
    with tsv.open("a") as f:
        if new:
            f.write(
                "stage\tprobe\tn\teffect_tvd\tmin_fisher_p\tcot_mention\tcot_mention_src\tcot_names_cued\tanswer_names_vendor\tnone\tqwen\tclaude\tgemini\n"
            )
        for e in summary:
            c = e["cells"]
            f.write(
                f"{stage}\t{e['probe']}\t{c['none']['n']}\t{e['effect']:.2f}\t{e['min_p']:.1e}\t{e['cot_mention']:.2f}\t{e['cot_mention_source']}\t"
                f"{e['cot_names_cued']:.2f}\t{e['answer_names_vendor']:.2f}\t"
                + "\t".join(json.dumps(c[k]["dist"]) if k in c else "" for k in ("none", "qwen", "claude", "gemini"))
                + "\n"
            )
