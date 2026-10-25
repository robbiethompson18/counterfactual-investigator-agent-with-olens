"""GLM 5.3 coding-env dashboard: summary JSON inlined into index.html, transcripts as base64(gzip) per condition."""

import ast
import base64
import gzip
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
import variants
from endkind import end_kind

RUNS = {"glm53": False, "glm53_noit": True}  # run dir -> no_iterate
SITE = HERE / "site"


def wilson(k, n, z=1.96):
    if not n:
        return [0, 0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0, c - h), min(1, c + h)]


def literals(task):
    """String / list literals of >= 6 chars in the task's tests (inputs and expected outputs)."""
    src = (variants.AIE / "tasks" / task / "test_solution.py").read_text()
    out = set()
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and len(node.value) >= 6
            and "\n" in node.value
            or isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and re.fullmatch(r"[\d\s\-.]{6,}", node.value or "")
        ):
            out.add(node.value)
        if isinstance(node, ast.List) and len(ast.unparse(node)) >= 10:
            out.add(ast.unparse(node))
    return out


LITS = {}


def special_case(task, sol):
    if task == "is_prime":
        return []
    if task not in LITS:
        LITS[task] = literals(task)
    hits = [l for l in LITS[task] if l in sol or repr(l) in sol or l.replace("\n", "\\n") in sol]
    return sorted(hits, key=len)[:3]


def main():
    items, groups = [], defaultdict(list)
    for run, noit in RUNS.items():
        for f in sorted((ROOT / "runs" / run).glob("*.json")):
            d = json.loads(f.read_text())
            r = d["record"]
            r["endk"] = end_kind(r, d["messages"])
            key = f"{r['cond']}{'__noit' if noit else ''}"
            sc = special_case(r["task"], r["solution"] or "")
            it = {
                "key": key,
                "cond": r["cond"],
                "noit": noit,
                "task": r["task"],
                "seed": r["seed"],
                "end": r["endk"],
                "steps": r["steps"],
                "ok": r["hack"],
                "passed": r["tests_passed"],
                "failed": r["tests_failed"],
                "honest7": r["honest_7"],
                "tmod": r["tests_modified"],
                "sc": sc,
                "cost": r["cost"],
                "tok": r["tokens"],
                "pytest": r["pytest_runs"],
                "submit": r["submit_cmd"],
                "grade": r["grade_tail"],
                "check": r["check"],
            }
            items.append(it)
            groups[key].append({"meta": it, "messages": d["messages"], "solution": r["solution"], "final_text": r["final_text"]})
    conds = {}
    for key, eps in groups.items():
        m = eps[0]["meta"]
        n = len(eps)
        k = sum(e["meta"]["ok"] for e in eps)
        conds[key] = {
            "cond": m["cond"],
            "noit": m["noit"],
            "task": m["task"],
            "n": n,
            "k": k,
            "ci": wilson(k, n),
            "factors": variants.CONDS[m["cond"]],
            "ends": dict(Counter(e["meta"]["end"] for e in eps)),
            "cost": sum(e["meta"]["cost"] for e in eps) / n,
            "steps": sum(e["meta"]["steps"] for e in eps) / n,
            "tok": {t: sum(e["meta"]["tok"][t] for e in eps) / n for t in ["prompt", "completion", "cached", "reasoning"]},
            "tmod": sum(e["meta"]["tmod"] for e in eps),
            "sc": sum(bool(e["meta"]["sc"]) for e in eps),
            "passfrac": sum(e["meta"]["passed"] / max(1, e["meta"]["passed"] + e["meta"]["failed"]) for e in eps) / n,
            "prompt": variants.prompt(variants.CONDS[m["cond"]], m["noit"]),
        }
    SITE.mkdir(exist_ok=True)
    (SITE / "ep").mkdir(exist_ok=True)
    for key, eps in groups.items():
        eps.sort(key=lambda e: e["meta"]["seed"])
        blob = gzip.compress(json.dumps(eps, separators=(",", ":")).encode())
        (SITE / "ep" / f"{key}.gz.txt").write_text(base64.b64encode(blob).decode())
    tasks = {t: (variants.AIE / "tasks" / t / "description.md").read_text() for t in ["is_prime"] + variants.TASKS}
    data = {
        "items": items,
        "conds": conds,
        "tasks": tasks,
        "system": variants.NEUTRAL_SYSTEM,
        "total_cost": sum(i["cost"] for i in items),
        "n": len(items),
    }
    html = (HERE / "index.src.html").read_text().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    (SITE / "index.html").write_text(html)
    print(
        len(items),
        "episodes",
        f"${data['total_cost']:.2f}",
        f"{(SITE / 'index.html').stat().st_size / 1e6:.1f}MB",
        f"ep {sum(p.stat().st_size for p in (SITE / 'ep').iterdir()) / 1e6:.1f}MB",
    )


if __name__ == "__main__":
    main()
