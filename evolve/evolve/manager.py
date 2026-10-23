"""Manager side of the agent job queue (see job_queue.py): a long-running Claude Code session claims jobs and runs
each one as a Claude Code subagent, so the Evolver and Creator draw from a Claude subscription instead of API
credits. These commands are what that session calls; MANAGER.md says how it uses them.

    python -m evolve.manager pending <run>                 job dirs nobody has claimed, oldest first
    python -m evolve.manager claim <job> [--by NAME]       atomically claim one job (exit 1 if already taken)
    python -m evolve.manager watch <run> [--timeout S]     block until an unclaimed job exists, then print it
    python -m evolve.manager finish <job> --agent-id ID    copy the subagent's transcript into the job; write
                                                           failed.json if it never submitted
    python -m evolve.manager fail <job> REASON             give up on a job
    python -m evolve.manager status <run>                  counts by state, and in-flight jobs with their age
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from .store import Store

CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"
TERMINAL = ("result.json", "failed.json")


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_new(path: Path, obj: dict) -> bool:
    """Create path with obj as JSON, only if it doesn't exist yet (O_EXCL, so two managers can't both claim)."""
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(obj) + "\n")
    return True


def jobs_dir(run: str) -> Path:
    return Store(run).dir / "agent_jobs"


def state(job: Path) -> str:
    for name, s in (("result.json", "done"), ("failed.json", "failed"), ("claimed.json", "claimed")):
        if (job / name).exists():
            return s
    return "pending" if (job / "job.json").exists() else "incomplete"  # job.json is written last by post()


def jobs(run: str) -> list[Path]:
    d = jobs_dir(run)
    if not d.is_dir():
        return []
    return sorted(
        (p for p in d.iterdir() if p.is_dir()), key=lambda p: (p / "job.json").stat().st_mtime if (p / "job.json").exists() else 0
    )


def pending(run: str) -> list[Path]:
    return [j for j in jobs(run) if state(j) == "pending"]


def claim(job: Path, by: str) -> bool:
    if state(job) != "pending":
        return False
    return _write_new(job / "claimed.json", {"by": by, "at": _now()})


def find_transcript(agent_id: str) -> Path | None:
    """The subagent's Claude Code transcript: ~/.claude/projects/<project>/<session>/subagents/agent-<id>.jsonl."""
    hits = sorted(CLAUDE_PROJECTS.glob(f"*/*/subagents/agent-{agent_id}.jsonl"), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def final_report(transcript: Path) -> str | None:
    """The subagent's hand-back message (SubagentHandback tool input), else its last assistant text."""
    handback, text = None, None
    for line in transcript.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("type") != "assistant":
            continue
        for c in (row.get("message") or {}).get("content") or []:
            if not isinstance(c, dict):
                continue
            if c.get("type") == "tool_use" and c.get("name") == "SubagentHandback":
                handback = (c.get("input") or {}).get("message") or handback
            elif c.get("type") == "text" and c.get("text", "").strip():
                text = c["text"]
    return handback or text


def finish(job: Path, agent_id: str | None, report: str | None) -> str:
    if agent_id:
        t = find_transcript(agent_id)
        if t:
            shutil.copy2(t, job / "transcript.jsonl")
            report = report or final_report(t)
        info = {"agent_id": agent_id, "transcript": str(t) if t else None, "finished": _now()}
        (job / "manager.json").write_text(json.dumps(info, indent=1) + "\n")
    if report:
        (job / "transcript.md").write_text(report.rstrip() + "\n")
    if not (job / "result.json").exists() and not (job / "failed.json").exists():
        _write_new(job / "failed.json", {"reason": "subagent ended without a passing submit", "at": _now()})
    copy_to_node(job)
    return state(job)


def copy_to_node(job: Path) -> Path | None:
    """The loop admits the child as soon as result.json appears, usually before finish runs, so copy the transcript
    files into the node it made (node.json "job" == this job's name) as agent_<name>, like loop.keep_session."""
    nodes = job.parent.parent / "nodes"
    for nj in nodes.glob("*/node.json"):
        try:
            if json.loads(nj.read_text()).get("job") != job.name:
                continue
        except (OSError, json.JSONDecodeError):
            continue
        for name in ("transcript.md", "transcript.jsonl", "manager.json"):
            if (job / name).exists():
                shutil.copy2(job / name, nj.parent / f"agent_{name}")
        return nj.parent
    return None


def main():
    ap = argparse.ArgumentParser(prog="python -m evolve.manager")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pending", "status"):
        sub.add_parser(name).add_argument("run")
    p = sub.add_parser("claim")
    p.add_argument("job", type=Path)
    p.add_argument("--by", default="claude-manager")
    p = sub.add_parser("watch")
    p.add_argument("run")
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--poll", type=float, default=5)
    p = sub.add_parser("finish")
    p.add_argument("job", type=Path)
    p.add_argument("--agent-id")
    p.add_argument("--report-file", type=Path, help="the subagent's final report, saved as transcript.md")
    p = sub.add_parser("fail")
    p.add_argument("job", type=Path)
    p.add_argument("reason")
    a = ap.parse_args()

    if a.cmd == "pending":
        for j in pending(a.run):
            print(j)
    elif a.cmd == "claim":
        if not claim(a.job.resolve(), a.by):
            print(f"not claimed: {a.job} is {state(a.job)}", file=sys.stderr)
            sys.exit(1)
        print(a.job.resolve())
    elif a.cmd == "watch":
        t0 = time.time()
        while time.time() - t0 < a.timeout:
            p = pending(a.run)
            if p:
                print("\n".join(map(str, p)))
                return
            time.sleep(a.poll)
        print("no pending jobs", file=sys.stderr)
        sys.exit(2)
    elif a.cmd == "finish":
        report = a.report_file.read_text() if a.report_file and a.report_file.exists() else None
        print(finish(a.job.resolve(), a.agent_id, report))
    elif a.cmd == "fail":
        if not _write_new(a.job / "failed.json", {"reason": a.reason, "at": _now()}):
            print("already failed", file=sys.stderr)
        print(state(a.job))
    elif a.cmd == "status":
        counts, live = {}, []
        for j in jobs(a.run):
            s = state(j)
            counts[s] = counts.get(s, 0) + 1
            if s == "claimed":
                c = json.loads((j / "claimed.json").read_text())
                kind = json.loads((j / "job.json").read_text()).get("kind")
                live.append(f"  {j.name} {kind} claimed {c.get('at')} by {c.get('by')}")
        print(" ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "no jobs")
        print("\n".join(live))


if __name__ == "__main__":
    main()
