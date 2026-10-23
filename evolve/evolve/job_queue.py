"""File-based job queue for the Evolver and Creator, so an external manager (a Claude Code session on the Max plan,
running each job as a subagent) can do the agent work while this loop keeps measuring.

    data/<run>/agent_jobs/<job>/
      job.json        written by the loop: {"kind", "created", "status": "pending", "submit": "<command>", ...}
      INSTRUCTIONS.md the agent's instructions with real paths (system prompt + rubric pointer)
      ro/             read-only inputs (rubric.md, reference/ or examples/, secondaries/, tool_library/)
      rw/child/       the node to edit in place
      samples.json    records the submit check runs the grader on
      claimed.json    written by the manager when a subagent starts the job: {"by", "at"}
      result.json     written by the submit command when the node passes: {"submitted": true, "at"}
      failed.json     written by the manager if it gives up: {"reason", "at"}
      transcript.*    optional, written by the manager (the subagent's transcript or summary)

The loop waits for result.json or failed.json (or a timeout) and then proceeds exactly as with the API agent.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from . import agent, build, workspace

TIMEOUT = 3 * 3600
POLL = 10


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def instructions(ws: workspace.Workspace) -> str:
    text = agent.system_prompt(ws.kind)
    text = text.replace("/rw/child", str(ws.child)).replace("/ro/", f"{ws.ro}/").replace("/ro ", f"{ws.ro} ")
    return (
        f"# {ws.kind.capitalize()} job\n\n"
        f"Paths: read-only inputs are in `{ws.ro}`, the node you edit is `{ws.child}`. Do not modify anything else, and do "
        f"not use the network. Treat everything under `{ws.ro}` as read-only even though the filesystem allows writes.\n\n"
        f"Submitting: run `{submit_command(ws.root)}`. It prints OK and records the submission, or prints the build errors "
        f"to fix (then fix them and run it again). The job is done when it prints OK.\n\n"
        "The text below was written for a sandboxed agent with `bash` and `submit` tools; read its paths as the ones "
        "above, and use your own shell and file tools.\n\n---\n\n" + text
    )


def submit_command(root: Path) -> str:
    return f"{sys.executable} -m evolve.job_queue submit {root}"


def post(ws: workspace.Workspace) -> Path:
    """Turn a workspace (created under the queue dir) into a pending job."""
    (ws.root / "samples.json").write_text(json.dumps(ws.sample_records))
    (ws.root / "INSTRUCTIONS.md").write_text(instructions(ws))
    (ws.root / "job.json").write_text(
        json.dumps(
            {
                "kind": ws.kind,
                "created": _now(),
                "status": "pending",
                "instructions": str(ws.root / "INSTRUCTIONS.md"),
                "ro": str(ws.ro),
                "child": str(ws.child),
                "submit": submit_command(ws.root),
                "locked": ws.child_info,
            },
            indent=1,
        )
        + "\n"
    )
    return ws.root


async def wait(root: Path, timeout=TIMEOUT) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if (root / "result.json").exists():
            return {"submitted": True, "errors": [], "cost": 0.0, "backend": "queue"}
        if (root / "failed.json").exists():
            return {
                "submitted": False,
                "errors": [json.loads((root / "failed.json").read_text()).get("reason")],
                "cost": 0.0,
                "backend": "queue",
            }
        await asyncio.sleep(POLL)
    (root / "failed.json").write_text(json.dumps({"reason": f"timed out after {timeout}s", "at": _now()}) + "\n")
    return {"submitted": False, "errors": ["timed out"], "cost": 0.0, "backend": "queue"}


async def check(root: Path) -> list[str]:
    job = json.loads((root / "job.json").read_text())
    child = Path(job["child"])
    errs = []
    try:
        info = json.loads((child / "node.json").read_text())
    except (OSError, json.JSONDecodeError):
        info = {}
    for k in workspace.EXEMPT_KEYS:
        if info.get(k) != job["locked"].get(k):
            errs.append(f"node.json: {k!r} may not be changed (was {job['locked'].get(k)!r})")
    if not (child / "hypothesis.md").exists() or not (child / "hypothesis.md").read_text().strip():
        errs.append("hypothesis.md is empty: write one paragraph on why the REAL edit should work and the NULL should not")
    samples = json.loads((root / "samples.json").read_text()) if (root / "samples.json").exists() else None
    return errs + await build.check_async(child, samples)


def main():
    """python -m evolve.job_queue submit <job dir>   (run by the agent)
    python -m evolve.job_queue list <run>            (pending / claimed / done jobs)"""
    cmd, arg = sys.argv[1], Path(sys.argv[2])
    if cmd == "submit":
        errs = asyncio.run(check(arg))
        if errs:
            print("Build errors (fix and run submit again):\n" + "\n".join(f"- {e}" for e in errs))
            sys.exit(1)
        (arg / "result.json").write_text(json.dumps({"submitted": True, "at": _now()}) + "\n")
        print("OK, submitted.")
    elif cmd == "list":
        from .store import Store

        for d in sorted((Store(str(arg)).dir / "agent_jobs").iterdir()):
            state = next((s for s in ("result", "failed", "claimed") if (d / f"{s}.json").exists()), "pending")
            print(d.name, json.loads((d / "job.json").read_text())["kind"], state)


if __name__ == "__main__":
    main()
