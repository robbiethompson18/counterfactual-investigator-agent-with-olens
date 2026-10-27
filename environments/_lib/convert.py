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
