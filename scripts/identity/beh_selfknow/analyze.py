"""Rank probes by identity sensitivity. Reads raw_<tag>.jsonl, prints tables, writes probes.json.

uv run --with scipy python scripts/identity/beh_selfknow/analyze.py [tag]
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scipy.stats import chi2_contingency, fisher_exact

sys.path.insert(0, "scripts/identity/beh_selfknow")
from probes import CONDITIONS, IDENTITY_OF, PROBES, SCORERS, VENDOR_RE, cot_self_ref

OUT = Path("data/runs/identity/2026-10-21_beh_selfknow")
tags = sys.argv[1].split(",") if len(sys.argv) > 1 else ["phase1", "phase2a", "phase2b", "phase2c", "phase3a", "phase3b", "phase3c"]
rows = []
for t in tags:
    rs = [json.loads(line) for line in (OUT / f"raw_{t}.jsonl").open()]
    jp = OUT / f"cot_judge_{t}.jsonl"
    judged = [json.loads(line)["cot_identity"] for line in jp.open()] if jp.exists() else [None] * len(rs)
    for r, j in zip(rs, judged):
        r["cot_identity"] = j
    rows += rs
by = defaultdict(list)
for r in rows:  # re-score from raw text so scorer fixes apply to old runs
    r["label"] = SCORERS[PROBES[r["probe"]][1]](r["content"])
    r["cot_self_ref"] = cot_self_ref(r["reasoning"])
    by[r["probe"], r["cond"]].append(r)


def dist(rs):
    c = Counter(r["label"] for r in rs)
    return {k: v / len(rs) for k, v in c.most_common()}


def tvd(a, b):
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in set(a) | set(b))


def chi_p(ra, rb):
    labels = sorted({r["label"] for r in ra + rb})
    if len(labels) < 2:
        return 1.0
    table = [[sum(r["label"] == lab for r in rs) for lab in labels] for rs in (ra, rb)]
    return float(chi2_contingency(table)[1])


def vendor_hit(label, vendor):
    return label == vendor or label == f"raw_http:{vendor}"


VENDOR_SCORED = {"first_sdk", "first_vendor", "self_vendor", "pip_vendor", "model_family", "url_vendor", "first_vendor_or_lib"}
summary = []
for p in PROBES:
    if (p, "none") not in by:
        continue
    base = by[p, "none"]
    entry = {"probe": p, "prompt": PROBES[p][0], "scorer": PROBES[p][1], "n_per_cell": len(base), "cells": {}}
    for c in CONDITIONS:
        rs = by.get((p, c))
        if not rs:
            continue
        cell = {
            "n": len(rs),
            "dist": {k: round(v, 3) for k, v in list(dist(rs).items())[:6]},
            "tvd_vs_none": round(tvd(dist(rs), dist(base)), 3),
            "chi2_p_vs_none": chi_p(rs, base),
            "cot_self_ref": round(sum(r["cot_self_ref"] for r in rs) / len(rs), 3),
            # judge (Qwen3.6-27B, thinking off, 3 replicates): which identity the CoT gives the assistant itself
            "cot_identity": {k: round(v / len(rs), 3) for k, v in Counter(r["cot_identity"] for r in rs).most_common()}
            if all(r["cot_identity"] for r in rs)
            else None,
            "truncated": sum(r["finish_reason"] != "stop" for r in rs),
        }
        v = IDENTITY_OF.get(c) or {
            "prime_anthropic": "anthropic",
            "prime_google": "google",
            "news_anthropic": "anthropic",
            "news_google": "google",
            "qwen_anthropic_partner": "anthropic",
        }.get(c)
        if v:  # strict non-silence: the CoT names the cued vendor/model anywhere (third-party mentions included)
            cell["cot_mentions_cued_vendor"] = round(sum(bool(VENDOR_RE[v].search(r["reasoning"] or "")) for r in rs) / len(rs), 3)
        if v and PROBES[p][1] in VENDOR_SCORED:
            k, kb = sum(vendor_hit(r["label"], v) for r in rs), sum(vendor_hit(r["label"], v) for r in base)
            hits = [r for r in rs if vendor_hit(r["label"], v)]
            cell |= {
                "target_vendor": v,
                "target_rate": round(k / len(rs), 3),
                "target_rate_none": round(kb / len(base), 3),
                "fisher_p": float(fisher_exact([[k, len(rs) - k], [kb, len(base) - kb]])[1]),
                # among replies that moved to the assigned vendor, share whose CoT never refers to its own identity
                "silent_share_of_hits": round(sum(not r["cot_self_ref"] for r in hits) / len(hits), 3) if hits else None,
                # stricter: judge says the CoT never states the assistant's own identity
                "judge_silent_share_of_hits": round(sum(r["cot_identity"] == "none" for r in hits) / len(hits), 3)
                if hits and all(r["cot_identity"] for r in hits)
                else None,
            }
        entry["cells"][c] = cell
    ids = [c for c in IDENTITY_OF if c in entry["cells"]]
    entry["mean_identity_tvd"] = round(sum(entry["cells"][c]["tvd_vs_none"] for c in ids) / len(ids), 3)
    pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1 :]]
    entry["max_pairwise_identity_tvd"] = round(max(tvd(dist(by[p, a]), dist(by[p, b])) for a, b in pairs), 3)
    if "helpful" in entry["cells"]:
        entry["priming_tvd_vs_helpful"] = {
            c: round(tvd(dist(by[p, c]), dist(by[p, "helpful"])), 3) for c in ("prime_anthropic", "prime_google") if (p, c) in by
        }
    entry["cot_self_ref_identity_mean"] = round(sum(entry["cells"][c]["cot_self_ref"] for c in ids) / len(ids), 3)
    summary.append(entry)

summary.sort(key=lambda e: -e["mean_identity_tvd"])
for rank, e in enumerate(summary, 1):
    e["rank"] = rank
for e in summary:
    print(
        f"\n== {e['probe']}  maxpairTVD={e['max_pairwise_identity_tvd']}  meanTVD={e['mean_identity_tvd']}  prime={e.get('priming_tvd_vs_helpful')}  cotself={e['cot_self_ref_identity_mean']}"
    )
    for c, cell in e["cells"].items():
        extra = (
            f" tgt={cell['target_rate']} (none {cell['target_rate_none']}) p={cell['fisher_p']:.1e} silent={cell['silent_share_of_hits']}"
            if "target_rate" in cell
            else ""
        )
        print(f"  {c:16s} cot={cell['cot_self_ref']:.2f} trunc={cell['truncated']} {cell['dist']}{extra}")
        if cell["cot_identity"]:
            print(f"  {'':16s} cot_identity={cell['cot_identity']} judge_silent={cell.get('judge_silent_share_of_hits')}")
doc = {
    "about": "beh_selfknow: probes whose answer depends on Qwen3.6-27B's assigned identity. Ranked by mean TVD of the answer-label "
    "distribution between each identity system prompt and no system prompt. Thinking on, T=0.6. cot_identity = Qwen3.6-27B judge "
    "(thinking off, 3 replicates, majority) of which identity the CoT gives the assistant itself. Raw rows: raw_<tag>.jsonl.",
    "tags": tags,
    "conditions": CONDITIONS,
    "probes": summary,
}
(OUT / "probes.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False))
print("\nwrote", OUT / "probes.json")
