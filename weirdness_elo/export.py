"""Merge run/ranking.json (from elo.py fit) with item fields -> run/ranking_full.json + run/ranking.csv, and per-item match reasons."""

import csv
import json
from pathlib import Path

RUN = Path(__file__).parent / "run"


def build():
    items = {it["id"]: it for it in json.loads((RUN / "items.json").read_text())}
    rk = (
        json.loads((RUN / "ranking.json").read_text())
        if (RUN / "ranking.json").exists()
        else {
            "summary": {},
            "ranking": [
                {"rank": i + 1, "id": k, "elo": 1500, "elo_sd": 0, "rank_90ci": [1, len(items)], "games": 0, "wins": 0}
                for i, k in enumerate(items)
            ],
        }
    )
    rows = []
    for r in rk["ranking"]:
        it = items[r["id"]]
        rows.append(
            {
                **r,
                "run": it["run"],
                "prompt_id": it["prompt_id"],
                "question": it["question"],
                "behavior": it["behavior"],
                "verification_score": it["verification_score"],
                "baseline_rate": it["baseline_rate"],
            }
        )
    # not yet judged: listed after the ranked items, rank/elo null
    ex = RUN / "items_extra.json"
    for it in json.loads(ex.read_text()) if ex.exists() else []:
        if it["id"] in items:
            continue
        rows.append(
            {
                "rank": None,
                "id": it["id"],
                "elo": None,
                "elo_sd": None,
                "rank_90ci": None,
                "games": 0,
                "wins": 0,
                "run": it["run"],
                "prompt_id": it["prompt_id"],
                "question": it["question"],
                "behavior": it["behavior"],
                "verification_score": it["verification_score"],
                "baseline_rate": it["baseline_rate"],
                "unranked": True,
            }
        )
    return {"summary": rk["summary"], "ranking": rows}


def matches():
    out = {}
    p = RUN / "comparisons.jsonl"
    if not p.exists():
        return out
    for l in p.read_text().splitlines():
        if not l.strip():
            continue
        c = json.loads(l)
        for me, other in ((c["a"], c["b"]), (c["b"], c["a"])):
            out.setdefault(me, []).append(
                {
                    "opponent": other,
                    "won": c["winner"] == me,
                    "first": c["first"] == me,
                    "margin": c.get("margin"),
                    "reason": c.get("reason"),
                    "round": c["round"],
                }
            )
    return out


if __name__ == "__main__":
    d = build()
    (RUN / "ranking_full.json").write_text(json.dumps(d, indent=1))
    with open(RUN / "ranking.csv", "w", newline="") as fh:
        cols = [
            "rank",
            "elo",
            "elo_sd",
            "rank_90ci_lo",
            "rank_90ci_hi",
            "games",
            "wins",
            "run",
            "prompt_id",
            "verification_score",
            "baseline_rate",
            "question",
            "behavior",
            "id",
        ]
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for r in (r for r in d["ranking"] if r["rank"] is not None):
            w.writerow({**{k: r[k] for k in cols if k in r}, "rank_90ci_lo": r["rank_90ci"][0], "rank_90ci_hi": r["rank_90ci"][1]})
    print(d["summary"])
