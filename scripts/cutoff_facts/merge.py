"""Merge the cutoff_facts subagents' per-domain JSONL files (BRIEF.md beside this script is what they were given) into the
cutoff_qa bank, `banks/cutoff_qa/v1.json` on the sprint HF dataset (banks are data: never in git).

    uv run python scripts/cutoff_facts/merge.py --pilot DIR/agent_*.jsonl --batch DIR/batch2/*.jsonl [--upload]

Pilot files (the 100-question first batch, 5 agents, snippet-verified) get ids prefixed `b1-` and `verified_by` set to
"batch1 (snippet)"; batch files are taken as written. Every row is checked against the brief's schema (exact keys, event_date
YYYY-MM within 2024-01..2026-09, verified_by, aliases a list) and the script prints: rows whose question doesn't end with
"Answer with just the <answer_type>.", questions that contain their own truth or an alias (leaks), and truths shared by
several items. It writes data/banks/cutoff_qa/v1.json (a header + `items`, each with `numeric` = the truth is digits) and
v1.md, a table of every item for reading. --upload puts v1.json and BRIEF.md under banks/cutoff_qa/ on HF.

Dedup was manual: same-event pairs and the two derivable answers were found by reading the table, and their ids are DROPPED
below (6 from the pilot's own cleaning, 22 from the merge), so the script reproduces the bank exactly. FIXES repairs three pilot
rows whose `domain` field held the question suffix.
"""

import argparse
import collections
import datetime as dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from environments.cutoff_qa.env import norm

REPO = "Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens"
OUT = ROOT / "data/banks/cutoff_qa"
KEYS = {
    "id",
    "domain",
    "event_date",
    "question",
    "answer_type",
    "truth",
    "truth_aliases",
    "pre_cutoff_analog",
    "source_url",
    "verified_by",
    "notes",
}
VERIFIED = {"fetched page", "two search results", "batch1 (snippet)"}
SUFFIX = "If you are not sure, give your best guess. Answer with just the {answer_type}."
NUMERIC = re.compile(r"^[\d.,% ]+$")
DROPPED = {  # manual (2026-10-29): same-event duplicates, keeping the better-phrased one, and two derivable answers
    "pilot": ["b1-space-10", "b1-space-20", "b1-tech-16", "b1-tech-18", "b1-sci-04", "b1-sci-20"],
    "merge": [
        "ai-12", "ai-36", "ai-39", "auction-23", "corp-01", "corp-22", "infra-02", "misc-32", "pc-37", "phone-09", "phone-44",
        "phone-46", "pol-42", "rank-11", "rank-18", "rank-21", "rank-28", "rank-45", "sci-26", "sci-31", "sw-16", "sw-32",
    ],
}  # fmt: skip
FIXES = {i: {"domain": "software and standards"} for i in ("b1-tech-05", "b1-tech-06", "b1-tech-15")}  # the pilot tech agent's bug
HEADER = {
    "family": "cutoff_qa",
    "version": 1,
    "written": "2026-10-29",
    "how": "Written by 25 Claude Sonnet subagents (a 100-question pilot over 5 domains, then 20 domains x 50) from Wikipedia "
    "'2024/2025/2026 in X' pages and web search, each answer verified on a fetched page or by two agreeing search results "
    "(the pilot: one result's snippet). Merged by scripts/cutoff_facts/merge.py; same-event duplicates and two derivable "
    "answers removed by hand (28 of 1,100).",
    "rules": "Each question pins one event by month and year and has one short unambiguous answer that is not derivable from "
    "the question; post-cutoff questions prefer a stale trap, recorded as pre_cutoff_analog. One question per record, office or "
    "annual list; later versions allow no numeric answers (v1 has some, flagged `numeric`).",
    "subject_cutoff": "Qwen3.6-27B, knowledge cutoff about January 2025; events 2024-01 .. 2026-09",
    "brief": "banks/cutoff_qa/BRIEF.md",
}


def load(paths: list[str], pilot: bool) -> list[dict]:
    rows = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if pilot:
                    r = {**r, "id": "b1-" + r["id"], "verified_by": "batch1 (snippet)"}
                rows.append(r)
    return rows


def check(r: dict) -> list[str]:
    bad = []
    if set(r) != KEYS:
        bad.append(f"keys {sorted(set(r) ^ KEYS)}")
    try:
        d = dt.date.fromisoformat(r["event_date"] + "-01")
        if not dt.date(2024, 1, 1) <= d <= dt.date(2026, 9, 1):
            bad.append(f"date out of range {r['event_date']}")
    except (ValueError, TypeError):
        bad.append(f"bad date {r.get('event_date')!r}")
    if r.get("verified_by") not in VERIFIED:
        bad.append(f"verified_by {r.get('verified_by')!r}")
    if not isinstance(r.get("truth"), str) or not r["truth"].strip():
        bad.append("empty truth")
    if not isinstance(r.get("truth_aliases"), list):
        bad.append("aliases not a list")
    if r.get("pre_cutoff_analog") is not None and not isinstance(r["pre_cutoff_analog"], str):
        bad.append("analog not a string")
    return bad


def leaks(r: dict) -> list[str]:
    q = norm(r["question"])
    return [t for t in [r["truth"], *r["truth_aliases"]] if len(norm(t)) >= 3 and re.search(rf"(?<!\S){re.escape(norm(t))}(?!\S)", q)]


COLUMNS = ["id", "event_date", "domain", "question", "truth", "truth_aliases", "pre_cutoff_analog", "verified_by", "source_url"]


def table(items: list[dict]) -> str:
    def cell(v) -> str:
        return ("; ".join(v) if isinstance(v, list) else str(v if v is not None else "")).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(cell(it[c]) for c in COLUMNS) + " |" for it in items]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pilot", nargs="*", default=[], help="the first batch's per-agent JSONL files (ids get b1-, verified_by batch1)")
    ap.add_argument("--batch", nargs="*", default=[], help="per-agent JSONL files taken as written")
    ap.add_argument("--upload", action="store_true", help="also upload v1.json and BRIEF.md to banks/cutoff_qa/ on HF")
    a = ap.parse_args()
    rows = load(a.pilot, pilot=True) + load(a.batch, pilot=False)
    ids = collections.Counter(r["id"] for r in rows)
    if dups := [i for i, n in ids.items() if n > 1]:
        sys.exit(f"duplicate ids across files: {dups}")
    dropped = set(DROPPED["pilot"]) | set(DROPPED["merge"])
    items = [{**r, **FIXES.get(r["id"], {})} for r in rows if r["id"] not in dropped]
    print(f"{len(rows)} rows in {len(a.pilot) + len(a.batch)} files, {len(dropped & set(ids))} dropped by hand -> {len(items)} items")
    for r in items:
        if bad := check(r):
            print(f"INVALID {r['id']}: {'; '.join(bad)}")
        if not r["question"].endswith(SUFFIX.format(answer_type=r["answer_type"])):
            print(f"SUFFIX {r['id']}: ...{r['question'][-60:]!r} (answer_type {r['answer_type']!r})")
        if hit := leaks(r):
            print(f"LEAK {r['id']}: truth {hit} appears in the question")
        r["numeric"] = bool(NUMERIC.match(r["truth"]))
    by_truth = collections.defaultdict(list)
    for r in items:
        by_truth[norm(r["truth"])].append(r["id"])
    shared = {t: ids for t, ids in by_truth.items() if len(ids) > 1}
    print(f"{len(shared)} truths shared by several items: " + "; ".join(f"{t} ({', '.join(v)})" for t, v in sorted(shared.items())))
    print(f"numeric {sum(1 for it in items if it['numeric'])}, with analog {sum(bool(it['pre_cutoff_analog']) for it in items)}, "
          f"by year {dict(sorted(collections.Counter(it['event_date'][:4] for it in items).items()))}, "
          f"verified {dict(collections.Counter(it['verified_by'] for it in items))}")  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v1.json").write_text(json.dumps({**HEADER, "n_items": len(items), "items": items}, indent=1, ensure_ascii=False) + "\n")
    (OUT / "v1.md").write_text(table(items))
    (OUT / "BRIEF.md").write_text((Path(__file__).with_name("BRIEF.md")).read_text())
    print(f"wrote {OUT}/v1.json, v1.md")
    if a.upload:
        from huggingface_hub import HfApi

        HfApi().upload_folder(
            repo_id=REPO, repo_type="dataset", folder_path=str(OUT), path_in_repo="banks/cutoff_qa",
            allow_patterns=["v1.json", "BRIEF.md"], commit_message=f"banks/cutoff_qa: v1, {len(items)} knowledge-cutoff questions",
        )  # fmt: skip
        print(f"uploaded -> https://huggingface.co/datasets/{REPO}/tree/main/banks/cutoff_qa")


if __name__ == "__main__":
    main()
