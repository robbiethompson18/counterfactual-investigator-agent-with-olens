"""Convert existing runs into run folders without rerunning anything, after checking that the ported env's build()
reproduces every stored episode's starting context byte for byte.

    uv run python -m environments.convert essay_grading     # data/runs/hidden_goal_grade/* -> data/envs/essay_grading/*
    uv run python -m environments.convert overeager         # the overeager rows -> data/envs/overeager_<task>/*

Converted runs keep each episode's Docent run id (`docent_run`) and the collection in meta.json (`docent_collection`), so
existing Docent links keep working, and meta.json says `converted_from: <old path>`. Their `measure` labels are the
stored detector / grader values (labels/measure@converted.jsonl); the check report also says where the ported measure()
disagrees with the stored values, when it can be recomputed from the row.
"""

import json
from pathlib import Path

from .core import episode_id
from .runfolder import RunWriter

DOCENT = "https://docent.transluce.org/dashboard"


def read_rows(run_dir: Path) -> tuple[list[dict], list]:
    rows = [json.loads(line) for line in (run_dir / "rows.jsonl").read_text().splitlines()]
    d = run_dir / "docent.json"
    doc = json.loads(d.read_text()) if d.exists() else {"collection_id": None, "runs": [None] * len(rows)}
    return rows, [doc["collection_id"], doc["runs"]]


def docent_url(coll, runs, i) -> str | None:
    return f"{DOCENT}/{coll}/agent_run/{runs[i]}" if coll and i < len(runs) and runs[i] else None


def first_diff(a: str, b: str) -> str:
    i = next((k for k, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
    return f"at char {i}: new {a[max(0, i - 40) : i + 40]!r} vs stored {b[max(0, i - 40) : i + 40]!r}"


def write_run(env, run_name, meta, episodes, labels):
    w = RunWriter(env, run_name, meta)
    for ep in episodes:
        w.add(ep)
    w.labels("measure", "converted", labels)
    w.finish(status=meta.get("source_status", "done"))
    return w.dir


def evalcue_target(meta: dict) -> dict:
    """A converted evalcue run's target, from its meta.json (model name and the harness's `served_by` string)."""
    m = meta["models"][0]
    served = m.get("served_by", "")
    raw = "raw completions" in served
    provider = None
    if served.startswith("openrouter (") and not raw:
        provider = served.split("(")[1].split(",")[0]
        provider = None if provider == "any provider" else provider
    backend = "openrouter-raw" if raw else "openrouter"
    return {
        "id": f"{m['name']}@{backend}:{provider or 'any'}",
        "model": m["name"],
        "backend": backend,
        "provider": provider,
        "thinking": meta["params"]["thinking"],
        "max_tokens": meta["params"].get("max_tokens"),
        "served_by": served,
    }


def convert_evalcue_runs(env, src: Path, dry=False) -> dict:
    """Single-turn evalcue runs -> run folders for `env` (a loaded env module): check build() against every row's
    prompt, keep Docent ids, store the old detector's values as labels (`positive` + its other fields)."""
    import sys

    report = {}
    for run in sorted(p for p in src.iterdir() if (p / "rows.jsonl").exists()):
        meta = json.loads((run / "meta.json").read_text())
        rows, (coll, docent_runs) = read_rows(run)
        mism, measure_diff, eps, labels = [], 0, [], []
        tgt = evalcue_target(meta)
        for i, r in enumerate(rows):
            ep = env.build(r["arm"], r["item"])
            r.setdefault("messages", [])  # an errored row (API failure) has no transcript
            for k, new in enumerate(ep.messages if r["messages"] else []):
                old = r["messages"][k]
                if new != {kk: old[kk] for kk in ("role", "content") if kk in old}:
                    mism.append(
                        {"line": i, "arm": r["arm"], "item": r["item"], "diff": first_diff(new["content"], old.get("content") or "")}
                    )
            eid = episode_id(r["arm"], r["item"], r["sample"])
            stored = {"positive": r.get("positive"), **(r.get("detect_notes") or {})}
            if "error" not in r and r.get("positive") is not None:
                recomputed = env.measure({"messages": r["messages"]})
                if any(recomputed.get(k) != v for k, v in stored.items() if k in recomputed):
                    measure_diff += 1
            eps.append(
                {
                    "id": eid, "arm": r["arm"], "item": r["item"], "sample": r["sample"], "target": tgt["id"],
                    "messages": r["messages"], "tools": [], "files_before": {}, "files_after": {}, "state": {},
                    "meta": r.get("meta", {}), "finish": r.get("finish"), "turns": r.get("turns"),
                    "usage": {"completion_tokens": r.get("completion_tokens")}, "error": r.get("error"),
                    "source": f"data/runs/{src.name}/{run.name}:{i}", "docent_run": docent_runs[i] if i < len(docent_runs) else None,
                    "docent_url": docent_url(coll, docent_runs, i),
                }
            )  # fmt: skip
            labels.append((eid, None if "error" in r else stored))
        report[run.name] = {
            "rows": len(rows),
            "start_context_mismatches": len(mism),
            "examples": mism[:3],
            "measure_disagrees": measure_diff,
        }
        if not dry and not mism:
            m = {
                "description": meta.get("description"),
                "converted_from": f"data/runs/{src.name}/{run.name}",
                "source_env": f"evalcue {src.name}",
                "git": meta.get("git_sha") or meta.get("git"),
                "command": meta.get("command"),
                "target": tgt,
                "params": meta.get("params"),
                "docent_collection": coll,
                "source_status": meta.get("status", "done"),
                "primary": env.PRIMARY,
            }
            write_run(env.NAME, run.name, m, eps, labels)
    print(json.dumps(report, indent=1), file=sys.stderr if dry else sys.stdout)
    return report
