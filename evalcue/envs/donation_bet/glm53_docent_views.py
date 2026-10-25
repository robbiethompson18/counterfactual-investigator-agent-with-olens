# /// script
# requires-python = ">=3.12"
# dependencies = ["docent-python>=0.1.88"]
# ///
"""Saved Docent filters for the GLM 5.3 Donation Bet report: one per (line set, arm), split by side of the line, and one
per (line set, pair) covering both sides. Writes glm53_docent_views.json (gitignored; the 2026-10-25 copy is on the HF dataset at
repo_data/evalcue/envs/donation_bet/glm53_docent_views.json), which glm53_report.py links from its tables.

A line set is picked out by the uploaded runs' `run_description` (the collection also holds other models' runs), so every
filter is: arm(s) AND run_description in the set's runs [AND positive]. Each filter's Docent count is checked against
the local rows.

  DOCENT_API_KEY=... uv run --script evalcue/envs/donation_bet/glm53_docent_views.py
"""

import json
from collections import Counter
from pathlib import Path

from docent import Docent  # ty: ignore[unresolved-import]  (PEP 723 script deps)

ENVD = Path(__file__).parent
RUNS = ENVD.parents[2] / "data/runs/donation_bet_goodbad"
OUT = ENVD / "glm53_docent_views.json"
D = "2026-10-23_glm53-"
# same tags as glm53_report.py
SETS = {
    "own": ["bets-ownline", "named-ownline", "bets-ownline-x4", "named-ownline-x4"],
    "qwen": ["bets", "named", "bets-x4", "named-x4"],
    "neutral": ["neutral", "neutral-x4"],
}


def eq(key, value):
    return {"type": "primitive", "key_path": ["metadata", *key.split(".")], "op": "==", "value": value}


def any_of(filters):
    return filters[0] if len(filters) == 1 else {"type": "complex", "op": "or", "filters": filters}


def main():
    c = Docent()
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    cid = None
    out = {"collection": None, "views": {}}
    for s, tags in SETS.items():
        descs, counts = [], Counter()
        for t in tags:
            run = RUNS / (D + t)
            meta, ids = json.loads((run / "meta.json").read_text()), json.loads((run / "docent.json").read_text())
            cid = cid or ids["collection_id"]
            assert ids["collection_id"] == cid
            descs.append(meta["description"])
            for r, i in zip(map(json.loads, (run / "rows.jsonl").open()), ids["runs"]):
                if i:
                    counts[r["arm"], "all"] += 1
                    counts[r["arm"], r.get("positive")] += 1
        runs_f = any_of([eq("run_description", d) for d in descs])
        arms = sorted({a for a, _ in counts})
        pairs = sorted({a.rsplit("_", 1)[0] for a in arms if a.endswith(("_above", "_below"))})
        specs = {}
        for a in arms:
            specs[f"{s}/{a}"] = ([eq("arm", a)], counts[a, "all"])
            for p in (True, False) if s != "neutral" else ():  # neutral rows have no positive label
                specs[f"{s}/{a}/{'le' if p else 'gt'}"] = ([eq("arm", a), eq("positive", p)], counts[a, p])
        for p in pairs:
            arms_p = [eq("arm", f"{p}_{side}") for side in ("below", "above")]
            specs[f"{s}/{p}"] = ([any_of(arms_p)], counts[f"{p}_below", "all"] + counts[f"{p}_above", "all"])
        for key, (conds, n) in specs.items():
            f = {"type": "complex", "op": "and", "filters": [*conds, runs_f]}
            v = old.get("views", {}).get(key)
            if not (v and v["filter"] == f):
                saved = c.create_saved_filter(cid, f, name=f"GLM 5.3 donation bet: {key}")
                v = {"url": saved["url"], "filter": f}
            v["n_local"] = n
            out["views"][key] = v
    out["collection"] = f"https://docent.transluce.org/dashboard/{cid}"
    # Docent-side counts per (run_description, arm, positive), to check the filters select what the local rows say
    res = c.dql_result_to_dicts(
        c.execute_dql(
            cid,
            "SELECT metadata_json->>'run_description' AS d, metadata_json->>'arm' AS arm, metadata_json->>'positive' AS pos,"
            " count(*) AS n FROM agent_runs GROUP BY 1, 2, 3",
        )
    )
    remote = Counter()
    for r in res:
        remote[r["d"], r["arm"], r["pos"]] += int(r["n"])
    bad = 0
    for s, tags in SETS.items():
        descs = {json.loads((RUNS / (D + t) / "meta.json").read_text())["description"] for t in tags}
        for key, v in out["views"].items():
            if not key.startswith(s + "/"):
                continue
            parts = key.split("/")
            arms = [parts[1]] if parts[1] in {a for _, a, _ in remote} else [f"{parts[1]}_below", f"{parts[1]}_above"]
            pos = None if len(parts) == 2 else ("true" if parts[2] == "le" else "false")
            n = sum(m for (d, a, p), m in remote.items() if d in descs and a in arms and (pos is None or p == pos))
            v["n_docent"] = n
            bad += n != v["n_local"]
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{len(out['views'])} views -> {OUT}; {bad} count mismatches")


if __name__ == "__main__":
    main()
