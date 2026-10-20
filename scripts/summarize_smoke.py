"""Summarize CHIVE smoke runs (partial runs are fine) into one JSON: per (eval, arm) the episodes
finished so far, free-form judge pass@1 next to Agam's Opus 5 draws on the same incidents, and
yes/no AUC / accuracy / Brier. Per-incident rows too.

Agam's per-draw judge scores come from his CHIVE cases viewer (climb_test = his 204 held-out
incidents, 5 draws each). Pass = judge score >= 7, as in his harness.

    uv run python scripts/summarize_smoke.py data/runs/chive data/runs/summary.json \\
        smoke_free:flash smoke_free_sonnet55:sonnet55 smoke_yesno:flash smoke_yesno_sonnet55:sonnet55
"""

import json
import sys
from pathlib import Path
from typing import Any

import requests

CMP_URL = "https://safety-sahan-2-agam--chive-cases-web.modal.run/data/v2/climb_test/cmp.json"
CMP = Path("data/viewer/cmp_climb_test.json")


def auc(pairs: list[tuple[float, bool]]) -> float | None:
    pos = [p for p, y in pairs if y]
    neg = [p for p, y in pairs if not y]
    if not pos or not neg:
        return None
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def pass_rate(scores: list[float]) -> float:
    return sum(s >= 7 for s in scores) / len(scores)


def yesno_metrics(evals: list[dict]) -> dict:
    per_inc = [a for e in evals if (a := auc([(it["p"], it["label"]) for it in e["items"]])) is not None]
    pooled = [(it["p"], it["label"]) for e in evals for it in e["items"]]
    return {
        "within_auc": mean(per_inc),
        "pooled_auc": auc(pooled),
        "accuracy": mean([e["accuracy"] for e in evals]),
        "brier": mean([e["brier"] for e in evals]),
        "n_items": len(pooled),
    }


root, out = Path(sys.argv[1]), Path(sys.argv[2])
if not CMP.exists():
    CMP.parent.mkdir(parents=True, exist_ok=True)
    CMP.write_text(requests.get(CMP_URL, timeout=120).text)
agam = {i["prompt_id"]: {a["arm"]: [s for _, s in a["draws"]] for a in i["arms"]} for i in json.loads(CMP.read_text())["incidents"]}
cells, rows = [], {}
for spec in sys.argv[3:]:
    ev, agent = spec.split(":")
    kind = "yesno" if "yesno" in ev else "free"
    for arm_dir in sorted((root / ev).glob("*/")):
        recs = [json.loads(f.read_text()) for f in sorted(arm_dir.glob("*.d0.json"))]
        done = [r for r in recs if r.get("result")]
        arm = arm_dir.name
        cell: dict[str, Any] = {
            "eval": kind,
            "agent": agent,
            "arm": arm,
            "done": len(done),
            "errors": sum(str(r.get("stopped_by", "")).startswith("error") for r in recs),
            "calls": mean([len(r.get("events", [])) for r in done]),
            "out_tokens": mean([r.get("output_tokens", 0) for r in done]),
            "seconds": mean([r.get("seconds", 0) for r in done]),
        }
        if kind == "free":
            judged = [r for r in done if (r.get("judge") or {}).get("score") is not None]
            cell["judged"] = len(judged)
            cell["pass1"] = pass_rate([r["judge"]["score"] for r in judged]) if judged else None
            cell["score"] = mean([r["judge"]["score"] for r in judged])
            # Agam only has draws on climb_test; on other splits this is None
            cell["agam_pass1"] = mean([pass_rate(agam[r["prompt_id"]][arm]) for r in judged if r["prompt_id"] in agam])
            for r in done:
                theirs = agam.get(r["prompt_id"])
                row = rows.setdefault(r["prompt_id"], {"agam": {a: theirs[a] for a in ("transcript", "olens_rl_B")} if theirs else None})
                row[f"{agent}/{arm}"] = (r.get("judge") or {}).get("score")
        else:
            evs = [r["yesno_eval"] for r in done if r.get("yesno_eval")]
            cell |= yesno_metrics(evs)
            committed = [r["yesno_text"] for r in done if r.get("yesno_text")]
            cell["committed"] = yesno_metrics(committed) if committed else None
        cells.append(cell)

out.write_text(json.dumps({"cells": cells, "rows": rows}, indent=1))
for c in cells:
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in c.items() if k != "committed"})
