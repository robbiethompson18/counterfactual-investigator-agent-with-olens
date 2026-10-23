"""Apply a node's grader.py to records, inside a sandbox (the grader is model-written code).

grade(record) must return True (behaviour shown), False (not shown) or None (unscorable: excluded from k and n).
Anything else, or an exception, counts as None with the error kept for the Rate auditor.
"""

import json

from .sandbox import Box

RUNNER = r"""
import json, sys, traceback
sys.path.insert(0, ".")
from grader import grade
out = []
for rec in json.load(sys.stdin):
    try:
        v = grade(rec)
        out.append(v if v is True or v is False or v is None else {"error": f"grade returned {v!r}"})
    except Exception:
        out.append({"error": traceback.format_exc(limit=3)[-800:]})
print("@@VERDICTS@@" + json.dumps(out))
"""


async def grade_records(grader_src: str, records: list[dict], image="evalcue-sandbox", timeout=None) -> list:
    if not records:
        return []
    timeout = timeout or 120 + 15 * len(records)  # graders may run tests per record
    async with Box({"grader.py": grader_src, "_run.py": RUNNER}, cwd="/grader", image=image) as box:
        code, out = await box.sh("python3 -I _run.py", timeout, stdin=json.dumps(records), max_out=0)
    if "@@VERDICTS@@" not in out:
        return [{"error": f"grader crashed (exit {code}): {out[-800:]}"}] * len(records)
    return json.loads(out.split("@@VERDICTS@@", 1)[1].strip().splitlines()[0])


def verdict(v):
    """True / False / None from a grade_records entry (errors are unscorable)."""
    return v if v in (True, False) else None
