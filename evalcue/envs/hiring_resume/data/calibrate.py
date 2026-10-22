"""Pick borderline bodies from a calibration run: per-body YES rate pooled over --arms, keep bodies whose rate is within
[LO, HI], and write calibrated.json (ordered so jobs interleave, since run_batch takes items round-robin).

    uv run python -m evalcue.envs.hiring_resume.data.calibrate data/runs/hiring_resume_calib_weak/DATE_TAG/rows.jsonl --arms base,bar_half
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("rows", type=Path)
ap.add_argument("--arms", default="base", help="comma-separated arms whose samples are pooled per body")
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
pool_arms = a.arms.split(",")


def pooled(item: str) -> float | None:
    v = [x for arm in pool_arms for x in rates[item].get(arm, [])]
    return sum(v) / len(v) if v else None


keep = [i for i in sorted(rates) if (x := pooled(i)) is not None and a.lo <= x <= a.hi]
by_job: dict[str, list[str]] = defaultdict(list)
for i in keep:
    by_job[i.rsplit("_", 1)[0]].append(i)
ordered = [ids[k] for k in range(max(map(len, by_job.values()))) for ids in by_job.values() if k < len(ids)]
print(f"kept {len(ordered)}/{len(rates)} bodies with pooled {a.arms} rate in [{a.lo}, {a.hi}]:", {j: len(v) for j, v in by_job.items()})
out = Path(__file__).with_name("calibrated.json")
out.write_text(json.dumps({"source": str(a.rows), "arms": pool_arms, "lo": a.lo, "hi": a.hi, "items": ordered}, indent=1) + "\n")
