"""Pick borderline bodies from a hiring_resume_calib run: per-body YES rate under the default context, keep bodies whose rate
is within [LO, HI], and write calibrated.json (ordered so jobs interleave, since run_batch takes items round-robin).

    uv run python -m evalcue.envs.hiring_resume_data.calibrate data/runs/hiring_resume_calib/DATE_TAG/rows.jsonl [--arm base]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("rows", type=Path)
ap.add_argument("--arm", default="base")
ap.add_argument("--lo", type=float, default=0.25)
ap.add_argument("--hi", type=float, default=0.75)
a = ap.parse_args()

rates: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
for line in a.rows.open():
    r = json.loads(line)
    if r.get("positive") is not None:
        rates[r["item"]][r["arm"]].append(r["positive"])


def rate(item: str, arm: str) -> float | None:
    v = rates[item].get(arm)
    return sum(v) / len(v) if v else None


arms = sorted({arm for v in rates.values() for arm in v})
for item in sorted(rates):
    print(item, "  ".join(f"{arm}={rate(item, arm):.2f}" if rate(item, arm) is not None else f"{arm}=NA" for arm in arms))
keep = [i for i in sorted(rates) if (x := rate(i, a.arm)) is not None and a.lo <= x <= a.hi]
by_job: dict[str, list[str]] = defaultdict(list)
for i in keep:
    by_job[i.rsplit("_", 1)[0]].append(i)
ordered = [ids[k] for k in range(max(map(len, by_job.values()))) for ids in by_job.values() if k < len(ids)]
print(f"kept {len(ordered)} bodies with {a.arm} rate in [{a.lo}, {a.hi}]:", {j: len(v) for j, v in by_job.items()})
out = Path(__file__).with_name("calibrated.json")
out.write_text(json.dumps({"source": str(a.rows), "arm": a.arm, "lo": a.lo, "hi": a.hi, "items": ordered}, indent=1) + "\n")
