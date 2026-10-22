"""Per-arm YES rates for hiring_resume runs, optionally restricted to the calibrated (borderline) bodies, with how often the
CoT mentions the edited attribute. Prints a Markdown table.

    uv run python -m evalcue.envs.hiring_resume_data.analyze data/runs/ENV/DATE_TAG/rows.jsonl [more rows.jsonl] [--calibrated]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evalcue.harness import wilson

ap = argparse.ArgumentParser()
ap.add_argument("rows", type=Path, nargs="+")
ap.add_argument("--calibrated", action="store_true", help="only bodies listed in calibrated.json")
a = ap.parse_args()

keep = None
if a.calibrated:
    keep = set(json.loads(Path(__file__).with_name("calibrated.json").read_text())["items"])
rows = [json.loads(line) for p in a.rows for line in p.open()]
rows = [r for r in rows if keep is None or r["item"] in keep or ("|" in r["item"] and set(r["item"].split("|")) <= keep)]
by_arm = defaultdict(list)
for r in rows:
    by_arm[r["arm"]].append(r)

print("| arm | YES k/n | rate | 95% CI | unscorable | CoT mentions edit | CoT demographic words | median tokens |")
print("|---|---|---|---|---|---|---|---|")
for arm, ar in by_arm.items():
    sc = [r for r in ar if r.get("positive") is not None]
    k = sum(r["positive"] for r in sc)
    lo, hi = wilson(k, len(sc))
    mention_keys = sorted({key for r in sc for key in r if key.startswith("mentions_") and key != "mentions_demographic"})
    edit_tokens = [m for m in mention_keys if not m.removeprefix("mentions_").startswith(("selective", "antibias", "eval", "deploy"))]
    mention = ", ".join(f"{m.removeprefix('mentions_')} {sum(r.get(m, False) for r in sc)}/{len(sc)}" for m in edit_tokens) or "-"
    demo = sum(r.get("mentions_demographic", False) for r in sc)
    toks = sorted(r.get("completion_tokens", 0) for r in ar)
    print(
        f"| {arm} | {k}/{len(sc)} | {k / max(len(sc), 1):.0%} | [{lo:.0%}, {hi:.0%}] | {len(ar) - len(sc)} | {mention} | "
        f"{demo}/{len(sc)} | {toks[len(toks) // 2] if toks else 0} |"
    )
