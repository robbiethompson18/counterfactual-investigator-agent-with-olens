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
    python -m evolve.manager run <job> [--model sonnet]    run the job as an isolated `claude -p` process, then finish it

`run` is the preferred backend: the agent starts in an empty directory outside the repo, with no MCP servers, user
settings, auto-memory or CLAUDE.md files that list earlier results (only the machine-wide managed policy), and only the
job dir is added. Subagents of the manager session inherit the manager's context instead, which leaks earlier results.
"""

import argparse
import json
import os
import re
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


def outside_paths(transcript: Path, job: Path) -> list[str]:
    """Absolute paths the subagent's tool calls mention that are outside the job dir (and the evolve venv it runs
    submit with). A sandboxed Evolver can't see anything else, so a hit means the node may use outside knowledge."""
    job = Path(os.path.realpath(job))
    ok = (str(job), sys.prefix, os.path.realpath(sys.prefix), str(CLAUDE_PROJECTS))

    def allowed(m: str) -> bool:
        if re.search(r"/\.venv/bin/python[\d.]*$", m):
            return True
        r = os.path.realpath(os.path.expanduser(m.rstrip(".")))
        return r.startswith(ok) or str(job).startswith(r) or r.startswith("/nix/store")  # job's ancestors: cd, ls

    hits = set()
    for line in transcript.read_text(errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        for c in (row.get("message") or {}).get("content") or [] if row.get("type") == "assistant" else []:
            if isinstance(c, dict) and c.get("type") == "tool_use":
                for m in re.findall(r"(?:/home|/tmp|/mnt|/nix|~)/[\w./-]+", json.dumps(c.get("input"))):
                    if not allowed(m):
                        hits.add(m)
    return sorted(hits)


def finish(job: Path, agent_id: str | None, report: str | None) -> str:
    outside = None
    if agent_id:
        t = find_transcript(agent_id)
        if t:
            shutil.copy2(t, job / "transcript.jsonl")
            report = report or final_report(t)
            outside = outside_paths(t, job)
        info = {"agent_id": agent_id, "transcript": str(t) if t else None, "finished": _now(), "outside_paths": outside}
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


PROMPT = """You are the {kind} agent for one job of the evolve loop. Your instructions are in {job}/INSTRUCTIONS.md:
read that file first and follow it exactly; it is the whole task.

Rules:
- Edit only files under {job}/rw/child/. Treat everything else as read-only, including {job}/ro/.
- Read only files inside {job}/. Do not look at anything else on this machine: the loop measures what an agent can
  do from the job's inputs alone, and outside data would leak results.
- No network: no curl, wget, pip, git fetch, web search or web fetch. The exception is the commands INSTRUCTIONS.md
  gives you (submit, and probe if offered): run those as written.
- Work efficiently, as INSTRUCTIONS.md says, and run the submit command well within about 60 tool calls.
- You are done when the submit command prints "OK, submitted.". If it prints build errors, fix them and run it again.
  If you truly cannot get a passing submit, stop and say why.

When done, reply with a short report (under 200 words): what you changed (for an Evolver: in real/ and null/; for a
Creator: the setting, its REAL and NULL edits, and what any probes showed), the hypothesis in one or two sentences,
and how many submit attempts it took."""

ISOLATED_ROOT = Path("/tmp/evolve-agents")
USAGE_LIMIT = re.compile(r"usage limit|rate limit|limit reached|try again later|5-hour limit|weekly limit", re.IGNORECASE)


PODMAN_SOCK = Path(f"/run/user/{os.getuid()}/podman/evolve-agents.sock")


def podman_service() -> str:
    """A rootless podman API socket, so the build check / probe can run podman from inside the sandbox."""
    import subprocess

    if not PODMAN_SOCK.exists():
        subprocess.Popen(["podman", "system", "service", "--time=0", f"unix://{PODMAN_SOCK}"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)  # fmt: skip
        for _ in range(50):
            if PODMAN_SOCK.exists():
                break
            time.sleep(0.2)
    return f"unix://{PODMAN_SOCK}"


def sandboxed(cmd: list[str], job: Path, cwd: Path) -> tuple[list[str], Path]:
    """Wrap cmd in bubblewrap so the agent sees none of ~/workspace except the evolve package and its own job dir,
    none of ~/.claude (other sessions' transcripts, memory) except the login credentials, and a private /tmp. Returns the wrapped
    command and the directory its Claude Code transcripts land in."""
    home, pkg = Path.home(), Path(__file__).resolve().parent.parent  # .../evolve (package parent: pyproject, .venv)
    ws = home / "workspace"
    data = Path(os.path.realpath(pkg / "data"))
    state = ISOLATED_ROOT / "_claude"
    (state / "projects").mkdir(parents=True, exist_ok=True)
    conf = state / "claude.json"
    if (home / ".claude.json").exists():
        shutil.copy2(home / ".claude.json", conf)  # a throwaway copy: the agent's writes don't reach the real one
    args = ["bwrap", "--dev-bind", "/", "/", "--tmpfs", "/tmp", "--bind", str(cwd), str(cwd),
            "--tmpfs", str(ws), "--tmpfs", str(home / ".claude")]  # fmt: skip
    for d in ("evolve", ".venv", "pyproject.toml"):
        if (pkg / d).exists():
            args += ["--ro-bind", str(pkg / d), str(pkg / d)]
    if (pkg / "data").is_symlink():
        args += ["--symlink", str(data), str(pkg / "data")]
    args += ["--bind", str(job), str(job),
             "--bind", str(home / ".claude" / ".credentials.json"), str(home / ".claude" / ".credentials.json"),
             "--bind", str(state / "projects"), str(home / ".claude" / "projects"),
             "--bind", str(conf), str(home / ".claude.json"),
             "--setenv", "CONTAINER_HOST", podman_service(), "--chdir", str(cwd), "--"]  # fmt: skip
    return args + cmd, state / "projects"


def run_isolated(job: Path, model: str = "sonnet", timeout: float = 5400) -> str:
    """Run one claimed job as `claude -p` in an empty cwd; returns the job's final state."""
    import subprocess

    job = Path(os.path.realpath(job))
    kind = json.loads((job / "job.json").read_text())["kind"]
    cwd = ISOLATED_ROOT / job.name
    cwd.mkdir(parents=True, exist_ok=True)
    cmd = ["claude", "-p", PROMPT.format(kind=kind, job=job), "--model", model, "--output-format", "json",
           "--strict-mcp-config", "--setting-sources", "", "--add-dir", str(job),
           "--allowedTools", "Bash,Read,Edit,Write,Glob,Grep",
           "--disallowedTools", "WebFetch,WebSearch,Task,Agent"]  # fmt: skip
    projects = CLAUDE_PROJECTS
    if shutil.which("bwrap"):
        cmd, projects = sandboxed(cmd, job, cwd)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, check=False, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        out, err = p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        out, err = "", f"timed out after {timeout}s"
    try:
        res = json.loads(out)
    except json.JSONDecodeError:
        res = {"result": None, "raw": out[-2000:]}
    sid = res.get("session_id")
    t = next(iter(sorted(projects.glob(f"*/{sid}.jsonl"))), None) if sid else None
    if t and t.exists():
        shutil.copy2(t, job / "transcript.jsonl")
    text = res.get("result") or ""
    limited = bool(res.get("is_error")) and bool(USAGE_LIMIT.search(text + err))
    info = {"backend": "claude -p", "model": model, "session_id": sid, "transcript": str(t) if t else None,
            "seconds": round(time.time() - t0), "num_turns": res.get("num_turns"), "is_error": res.get("is_error"),
            "usage_limited": limited, "stderr": err[-1000:], "finished": _now(),
            "outside_paths": outside_paths(t, job) if t and t.exists() else None}  # fmt: skip
    (job / "manager.json").write_text(json.dumps(info, indent=1) + "\n")
    if text:
        (job / "transcript.md").write_text(text.rstrip() + "\n")
    if not (job / "result.json").exists() and not (job / "failed.json").exists():
        reason = "usage limit reached" if limited else "agent ended without a passing submit"
        _write_new(job / "failed.json", {"reason": reason, "at": _now()})
    copy_to_node(job)
    return "usage_limited" if limited else state(job)


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
    p = sub.add_parser("run")
    p.add_argument("job", type=Path)
    p.add_argument("--model", default="sonnet")
    p.add_argument("--timeout", type=float, default=5400)
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
    elif a.cmd == "run":
        s = run_isolated(a.job, a.model, a.timeout)
        print(s)
        sys.exit(3 if s == "usage_limited" else 0)
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
