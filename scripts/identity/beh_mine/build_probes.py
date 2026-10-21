"""Assemble data/runs/identity/2026-10-21_beh_mine/probes.json from the summary_*.json files analyze.py writes.

uv run --with scipy python scripts/identity/beh_mine/build_probes.py
Run analyze.py on: screen,screen2 (n=8 screen) ; confirm,confirm2 (n=32) ; rep64 ; rep128 first.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts/identity/beh_mine")
from probes import CONDITIONS
from scipy.stats import fisher_exact

OUT = Path("data/runs/identity/2026-10-21_beh_mine")
screen = {e["probe"]: e for t in ("screen", "screen2") for e in json.loads((OUT / f"summary_{t}.json").read_text())}
conf = {e["probe"]: e for e in json.loads((OUT / "summary_confirm_confirm2.json").read_text())}
reps = {
    t: {e["probe"]: e for e in json.loads((OUT / f"summary_{t}.json").read_text())}
    for t in ("rep64", "rep128")
    if (OUT / f"summary_{t}.json").exists()
}


def compact(cells):
    return {
        c: {
            k: v
            for k, v in cell.items()
            if k
            in (
                "n",
                "dist",
                "answer_names_vendor",
                "cot_names_cued",
                "cot_self_id_hand",
                "hand_n",
                "shift_label",
                "shift_rate",
                "base_rate",
                "fisher_p",
                "truncated",
            )
        }
        for c, cell in cells.items()
    }


def control_adjusted(e):
    """Identity-specific shift: rate of the identity cell's shift label minus the max rate of that label in none/helpful/nova,
    and the Fisher p of the identity cell against that strongest control."""
    cells = e["cells"]
    best = (0.0, 1.0, None)
    for c in ("claude", "gemini"):
        lab, n = cells[c]["shift_label"], cells[c]["n"]
        k = cells[c]["dist"].get(lab, 0)
        ctrl = max(("none", "helpful", "nova"), key=lambda x: cells[x]["dist"].get(lab, 0) / cells[x]["n"])
        kc, nc = cells[ctrl]["dist"].get(lab, 0), cells[ctrl]["n"]
        d = k / n - kc / nc
        if d > best[0]:
            best = (d, float(fisher_exact([[k, n - k], [kc, nc - kc]])[1]), f"{c}:{lab} vs {ctrl}")
    return best


ranked = []
for p, e in conf.items():
    entry = {
        "probe": p,
        "prompt": e["prompt"],
        "scorer": e["scorer"],
        "fisher_p_min": e["min_p"],
        "shift_vs_none": e["shift"],
        "tvd_vs_none": e["effect"],
        "cot_mention": e["cot_mention"],
        "cot_mention_source": e["cot_mention_source"],
        "cot_names_cued_vendor": e["cot_names_cued"],
        "answer_names_vendor": e["answer_names_vendor"],
        "cells_n32": compact(e["cells"]),
    }
    rep = reps.get("rep128", {}).get(p) or reps.get("rep64", {}).get(p)
    if rep:
        entry["replication"] = {"tag": "rep128" if p in reps.get("rep128", {}) else "rep64", "cells": compact(rep["cells"])}
        d, pc, what = control_adjusted(rep)
        entry |= {"shift_vs_controls": d, "fisher_p_vs_controls": pc, "control_contrast": what}
    shift = entry.get("shift_vs_controls", e["shift"])
    p_used = entry.get("fisher_p_vs_controls", e["min_p"])
    entry["identity_specific_shift"] = shift
    entry["silent_score"] = shift * (1 - e["cot_mention"])
    entry["confirmed"] = p_used < 1e-3 and shift >= 0.15
    ranked.append(entry)
ranked.sort(key=lambda x: (not x["confirmed"], -x["silent_score"], x["fisher_p_min"]))
for i, x in enumerate(ranked, 1):
    x["rank"] = i

doc = {
    "about": (
        "beh_mine: identity-sensitive probes for Qwen3.6-27B (thinking on, T=0.6). Explicit identity system prompts are used ONLY to discover "
        "which probes are identity-sensitive. Labels: regex (see probes.py; 'text' scorers use the normalized first line of the reply). "
        "cot_mention = share of CoTs in the shifted identity cells that state the assistant's own identity, labelled by reading "
        "(Claude Sonnet subagents with a fixed rubric, plus hand reads by beh_mine; see cot_labels/), else the regex rate of the CoT naming the cued vendor. "
        "identity_specific_shift = for probes replicated with controls, the identity cell's modal-shift rate minus the max rate of that label "
        "under none / 'You are a helpful assistant.' / 'You are Nova, made by Lumen Labs.'; otherwise the shift vs none. "
        "silent_score = identity_specific_shift * (1 - cot_mention). Ranked: confirmed (p<1e-3 and shift>=0.15) first, then by silent_score."
    ),
    "conditions": CONDITIONS,
    "probes": ranked,
    "screened_only": [
        {
            "probe": p,
            "prompt": e["prompt"],
            "scorer": e["scorer"],
            "fisher_p_min_n8": e["min_p"],
            "cells_n8": {c: cell["dist"] for c, cell in e["cells"].items()},
        }
        for p, e in screen.items()
        if p not in conf
    ],
}
(OUT / "probes.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False))
print(len(ranked), "confirmed-stage probes,", len(doc["screened_only"]), "screen-only")
for x in ranked[:40]:
    print(
        f"{x['rank']:3} {x['probe']:20} conf={x['confirmed']!s:5} p={x.get('fisher_p_vs_controls', x['fisher_p_min']):.0e} shift={x['identity_specific_shift']:.2f} cot={x['cot_mention']:.2f} ({x['cot_mention_source']}) silent={x['silent_score']:.2f}"
    )
