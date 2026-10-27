# /// script
# requires-python = ">=3.12"
# dependencies = ["docent-python>=0.1.88", "huggingface_hub"]
# ///
"""Mirror run dirs into Docent (https://docent.transluce.org) as public collections, so every transcript has a link.

evalcue-harness runs (rows.jsonl) go to one collection per env, cfbench runs (predictions.jsonl) to "cfbench". Each upload
writes docent.json into the run dir: {"collection_id", "runs": [Docent agent run id for each line, null for skipped rows]}.
runs.finish() runs this before the HF sync, so docent.json ships with the run. Docent's SDK pulls ~90 packages, so this
script carries its own deps (PEP 723) instead of adding them to the project: always run it with `uv run --script`.

    uv run --script scripts/docent_upload.py upload RUN [RUN ...]   # skips runs that already have docent.json
    uv run --script scripts/docent_upload.py backfill               # every local run dir without docent.json
    uv run --script scripts/docent_upload.py link ENV/RUN:LINE      # cfbench's `shown` format, LINE 0-based
    uv run --script scripts/docent_upload.py view ENV arm=X positive=true   # saved filter -> shareable filtered-table URL

Needs DOCENT_API_KEY (in .envrc.local).
"""

import argparse
import json
import sys
from functools import cache
from pathlib import Path

from docent import Docent  # ty: ignore[unresolved-import]  (PEP 723 script deps)
from docent.data_models import AgentRun, Transcript  # ty: ignore[unresolved-import]  (PEP 723 script deps)
from docent.data_models.chat import AssistantMessage, ContentReasoning, ContentText, SystemMessage, ToolCall, ToolMessage, UserMessage  # ty: ignore[unresolved-import]  (PEP 723 script deps)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # not installed in this script's own env
from counterfactual_investigator_agent_with_olens.runs import hf_upload

ROOT = Path("data/runs")
WEB = "https://docent.transluce.org"
# rows.jsonl keys that aren't per-env behaviour flags
STRUCTURAL = {"env", "model", "arm", "item", "sample", "seed", "thinking", "messages", "meta", "files_before", "files_after", "cwd"}
FILE_CHARS = 5_000


@cache
def client() -> Docent:
    return Docent()


@cache
def collections() -> dict[str, str]:
    return {c["name"]: c["id"] for c in client().list_collections()}


def collection_id(name: str) -> str:
    """Find the collection by name, else create it public. Looked up by name so checkouts and machines share one."""
    if name not in collections():
        cid = client().create_collection(name=name, description="Counterfactual-investigator sprint (Robbie & Kylie, MATS)")
        client().make_collection_public(cid)
        collections()[name] = cid
    return collections()[name]


def messages(raw: list[dict]) -> list:
    tool_names = {}  # tool_call_id -> function name, so a tool result can say which tool it answers
    out = []
    for m in raw:
        role, content = m["role"], m.get("content") or ""
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "user":
            out.append(UserMessage(content=content))
        elif role == "tool":
            out.append(ToolMessage(content=content, tool_call_id=m.get("tool_call_id"), function=tool_names.get(m.get("tool_call_id"))))
        else:
            parts = []
            if reasoning := m.get("reasoning_content") or m.get("reasoning"):  # vLLM / OpenRouter spellings
                parts.append(ContentReasoning(reasoning=reasoning))
            if content:
                parts.append(ContentText(text=content))
            calls = []
            for tc in m.get("tool_calls") or []:
                try:
                    args = json.loads(tc["function"]["arguments"])
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": tc["function"]["arguments"]}
                tool_names[tc["id"]] = tc["function"]["name"]
                calls.append(ToolCall(id=tc["id"], function=tc["function"]["name"], arguments=args, type="function"))
            out.append(AssistantMessage(content=parts or "", tool_calls=calls or None))
    return out


def changed_files(row: dict) -> str | None:
    before, after = row.get("files_before") or {}, row.get("files_after") or {}
    changed = {path: body for path, body in sorted(after.items()) if before.get(path) != body}
    return "\n\n".join(f"--- {path}\n{body[:FILE_CHARS]}" for path, body in changed.items()) or None


def env_row_run(row: dict, source: str) -> AgentRun:
    # coding-harness rows (ImpossibleBench / realism) name these variant / pid / seed instead of arm / item / sample
    row = {**row, "arm": row.get("arm", row.get("variant")), "item": row.get("item", row.get("pid", row.get("task")))}
    row["sample"] = row.get("sample", row.get("seed"))
    metadata = {
        **{k: v for k, v in row.items() if k not in STRUCTURAL},  # the env's behaviour flags, `positive`, `finish`, ...
        "arm": row["arm"],
        "model": row.get("model"),
        "item": row["item"],
        "sample": row["sample"],
        "seed": row.get("seed"),
        "thinking": row.get("thinking"),
        "source": source,
        "env_meta": row.get("meta"),
        "changed_files": changed_files(row),
    }
    return AgentRun(name=f"{row['arm']} #{row['sample']}", transcripts=[Transcript(messages=messages(row["messages"]))], metadata=metadata)


def link_for(source: str) -> str | None:
    """`env/run:line` -> that row's Docent link, if its run has been uploaded locally."""
    run, _, line = source.rpartition(":")
    index = ROOT / run / "docent.json"
    if not index.exists():
        return None
    meta = json.loads(index.read_text())
    run_id = meta["runs"][int(line)]
    return run_id and f"{WEB}/dashboard/{meta['collection_id']}/agent_run/{run_id}"


def prediction_run(row: dict, source: str) -> AgentRun:
    metadata = {k: v for k, v in row.items() if k != "messages"}
    metadata |= {"source": source, "shown_links": [link_for(s) for s in row.get("shown", [])]}
    name = f"{row['base_arm']} -> {row['cf_arm']} ({row['condition']}, {row['predictor']})"
    return AgentRun(name=name, transcripts=[Transcript(messages=messages(row["messages"]))], metadata=metadata)


def upload(run: Path) -> bool:
    """Upload one run dir and write its docent.json; False if it has nothing Docent-shaped or was already uploaded."""
    if (run / "docent.json").exists():
        return False
    meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {"status": "running"}
    if meta["status"] == "running":
        return False  # still being written; its runs.finish() uploads it
    if (run / "rows.jsonl").exists():
        lines, convert, name = (run / "rows.jsonl").read_text().splitlines(), env_row_run, run.parent.name
    elif (run / "predictions.jsonl").exists():
        lines, convert, name = (run / "predictions.jsonl").read_text().splitlines(), prediction_run, "cfbench"
    else:
        return False
    rows = [json.loads(line) for line in lines]
    if run.parent.name != "cfbench" and not all("arm" in r or "variant" in r for r in rows):
        return False  # a rows.jsonl from something other than the evalcue harness
    agent_runs = [convert(r, f"{run.parent.name}/{run.name}:{i}") if r.get("messages") else None for i, r in enumerate(rows)]
    for agent_run in filter(None, agent_runs):
        agent_run.metadata |= {"run_status": meta["status"], "run_description": meta.get("description")}
    cid = collection_id(name)
    if any(agent_runs):
        client().add_agent_runs(cid, [r for r in agent_runs if r])
    (run / "docent.json").write_text(json.dumps({"collection_id": cid, "runs": [r and r.id for r in agent_runs]}) + "\n")
    print(f"{run}: {sum(map(bool, agent_runs))}/{len(rows)} rows -> {WEB}/dashboard/{cid}")
    return True


def view(collection: str, conditions: list[str]) -> str:
    """Save a filter ANDing `key=value` metadata conditions (values parsed as JSON when they can be) and return its URL."""
    filters = []
    for condition in conditions:
        key, _, value = condition.partition("=")
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        filters.append({"type": "primitive", "key_path": ["metadata", *key.split(".")], "op": "==", "value": value})
    if collection not in collections():
        sys.exit(f"no Docent collection named {collection!r}")
    saved = client().create_saved_filter(
        collections()[collection], {"type": "complex", "op": "and", "filters": filters}, name=" ".join(conditions)
    )
    return saved["url"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("upload").add_argument("runs", type=Path, nargs="+")
    sub.add_parser("backfill")
    sub.add_parser("link").add_argument("source", help="ENV/RUN:LINE, e.g. shutdown_palisade/2026-10-22_r15_catch_words:3")
    p = sub.add_parser("view")
    p.add_argument("collection", help="env name, or cfbench")
    p.add_argument("conditions", nargs="+", help="KEY=VALUE metadata conditions, e.g. arm=catch_bydefault_D positive=true")
    args = ap.parse_args()
    if args.cmd == "link":
        print(link_for(args.source) or sys.exit(f"{args.source}: run not uploaded (no docent.json)"))
        return
    if args.cmd == "view":
        print(view(args.collection, args.conditions))
        return
    if args.cmd == "upload":
        for run in args.runs:
            upload(run)
        return
    # cfbench last: its predictions link to the env rows they showed the predictor
    uploaded = [upload(run) for run in sorted(ROOT.glob("*/*/"), key=lambda p: (p.parent.name == "cfbench", p))]
    if any(uploaded):
        # one HF commit for every new docent.json, rather than one per run as runs.py's sync would make
        hf_upload(ROOT, "runs", "runs/*/*/docent.json: Docent backfill", allow_patterns="*/*/docent.json")


if __name__ == "__main__":
    main()
