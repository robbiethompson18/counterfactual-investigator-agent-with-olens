"""Where the Qwen3.6-27B items land in run/ranking.json."""

import json
import statistics
from pathlib import Path

r = json.loads(Path("run/ranking.json").read_text())
it = {i["id"]: i for i in json.loads(Path("run/items.json").read_text())}
f = set(json.loads(Path("run/focus_qwen27b.json").read_text()))
q = [x for x in r["ranking"] if x["id"] in f]
print(json.dumps(r["summary"]))
print(
    {
        "n": len(q),
        "median_rank": statistics.median(x["rank"] for x in q),
        "top100": sum(x["rank"] <= 100 for x in q),
        "median_games": statistics.median(x["games"] for x in q),
        "of": len(r["ranking"]),
    }
)
for x in q[:5]:
    print(x["rank"], x["elo"], "±", x["elo_sd"], x["games"], it[x["id"]]["run"], "|", it[x["id"]]["behavior"][:160])
