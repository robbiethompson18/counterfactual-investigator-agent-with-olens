"""Export a sample of is_prime episodes + condition definitions for the transcript viewer."""

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import port
from endkind import end_kind


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h)), round(100 * (c + h))]


def clip(s, n):
    if s is None:
        return None
    return s if len(s) <= n else s[: n // 2] + f"\n…[{len(s) - n} chars clipped]…\n" + s[-n // 2 :]


def outcome(r):
    if r["hack"]:
        return "hack"
    if r["honest_7"]:
        return "honest"
    return "other"


MODELS = {
    "ns_v4f": "DeepSeek V4 Flash",
    "ns_qwen": "Qwen3.6-27B",
    "noit_v4f": "V4 Flash · no pytest line",
    "noit_qwen": "Qwen · no pytest line",
}
PER = 5  # per outcome per condition
random.seed(0)
out = {"models": {}, "conditions": {}, "task": {}}
aie = port.AIE / "tasks/is_prime"
out["task"] = {f: (aie / f).read_text() for f in ["description.md", "solution.py", "test_solution.py"]}
out["task"]["submit_solution.py"] = port.SUBMIT_SCRIPTS["reward"].read_text()
out["task"]["submit_solution_counts.py"] = port.SUBMIT_SCRIPTS["counts"].read_text()
for k in ("fractional", "self_report"):
    out["task"][f"submit_solution_{k}.py"] = port.SUBMIT_SCRIPTS[k].read_text()
out["systems"] = {k: port.NEUTRAL_SYSTEM for k in MODELS}
for name, (tools, has_submit, ending, reply_ends) in port.CONDS.items():
    if not reply_ends or name in (
        "submit_reply",
        "submitC_reply",
        "submitR_short_reply",
        "submitC_short_reply",
        "submitR_nosec_reply",
        "submitC_nosec_reply",
    ):
        continue  # superseded "submit your work" wording
    sl = "- submit_solution.py: Submission script\n" if has_submit else ""
    out["conditions"][name] = {
        "tools": [t["function"] for t in tools],
        "submit_script": has_submit,
        "counts": has_submit == "counts",
        "script_kind": has_submit or None,
        "reply_ends": reply_ends,
        "system": port.SYSTEM,
        "user": port.PROMPT_HEAD.format(submit_line=sl) + ending,
    }
out["reprompt"] = port.REPROMPT
for run, label in MODELS.items():
    by = defaultdict(list)
    for f in sorted(Path("runs", run).glob("*.json")):
        by[json.loads(Path(f).read_text())["record"]["cond"]].append(f)
    m = {"label": label, "conds": {}}
    for cond, files in by.items():
        if cond not in out["conditions"]:
            continue
        recs = [(f, json.loads(Path(f).read_text())) for f in files]
        for _, d in recs:
            d["record"]["end"] = end_kind(d["record"], d["messages"])
        k = sum(d["record"]["hack"] for _, d in recs)
        n = len(recs)
        groups = defaultdict(list)
        for f, d in recs:
            groups[outcome(d["record"])].append(d)
        eps = []
        for oc, ds in groups.items():
            for d in random.sample(ds, min(PER, len(ds))):
                r = d["record"]
                msgs = []
                for x in d["messages"][2:]:
                    if x["role"] == "assistant":
                        msgs.append(
                            {
                                "role": "assistant",
                                "reasoning": clip(x.get("reasoning"), 6000),
                                "content": clip(x.get("content"), 6000),
                                "calls": [
                                    {"name": c["function"]["name"], "args": clip(c["function"].get("arguments"), 4000)}
                                    for c in x.get("tool_calls") or []
                                ],
                            }
                        )
                    else:
                        msgs.append(
                            {
                                "role": x["role"],
                                "content": clip(
                                    x.get("content") if isinstance(x.get("content"), str) else json.dumps(x.get("content")), 2500
                                ),
                            }
                        )
                eps.append(
                    {
                        "seed": r["seed"],
                        "outcome": oc,
                        "end": r["end"],
                        "steps": r["steps"],
                        "check": r["check"],
                        "solution": r["solution"],
                        "messages": msgs,
                    }
                )
        counts = {oc: len(ds) for oc, ds in groups.items()}
        ends = defaultdict(lambda: defaultdict(int))
        for _, d in recs:
            ends[outcome(d["record"])][d["record"]["end"]] += 1
        ends = {k: dict(v) for k, v in ends.items()}
        sl = "- submit_solution.py: Submission script\n" if port.CONDS[cond][1] else ""
        head = port.PROMPT_HEAD.replace("\n" + port.ITERATE_LINE, "") if run.startswith("noit") else port.PROMPT_HEAD
        m["conds"][cond] = {
            "user": head.format(submit_line=sl) + port.CONDS[cond][2],
            "n": n,
            "hack": k,
            "pct": round(100 * k / n),
            "ci": wilson(k, n),
            "counts": counts,
            "ends": dict(ends),
            "episodes": sorted(eps, key=lambda e: (e["outcome"], e["seed"])),
        }
    out["models"][run] = m
Path("viewer/data.json").write_text(json.dumps(out, separators=(",", ":")))
print(Path("viewer/data.json").stat().st_size / 1e6, "MB")
