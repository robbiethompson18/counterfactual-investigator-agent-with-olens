"""Evolver / Creator agent session: a tool loop (bash + submit) over a podman Box with /ro and /rw mounted.

    res = await run_agent("evolver", ws.ro, ws.rw, sample_records=ws.sample_records, locked=ws.child_info)

bash runs in one Box for the whole session (no network); submit runs build.check on /rw/child and ends the
session when it passes. Files the container writes into /rw are handed back to the host user afterwards.
"""

import json
import shutil
import time
from pathlib import Path

from . import build, llm
from .sandbox import Box, run
from .workspace import EXEMPT_KEYS

PROMPTS = Path(__file__).resolve().parent / "prompts"
CMD_TIMEOUT = 60
MAX_OUT = 10000
MARKER = "<!-- evolver_mode"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command in the container. Each call starts a fresh shell in /rw/child. "
            f"{CMD_TIMEOUT} s timeout; long output is truncated.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Check the node in /rw/child and submit it. Returns 'OK, submitted.' (the session ends) or "
            "a list of build errors to fix before submitting again.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def system_prompt(kind: str) -> str:
    text = (PROMPTS / "agent_system.md").read_text()
    common, _, evolver = text.partition(MARKER)
    evolver = evolver.split("-->", 1)[1].strip() if evolver else ""
    mode = (PROMPTS / "creator_mode.md").read_text().strip() if kind == "creator" else evolver
    return common.strip().replace("{mode}", mode) + "\n"


KICKOFF = {
    "evolver": "Improve the reference node. Start by reading /ro/rubric.md, then the reference's results.",
    "creator": "Invent a new setting. Start by reading /ro/rubric.md, then look at an example or two.",
}


def _cached(messages: list[dict]) -> list[dict]:
    """Copy with Anthropic prompt-cache breakpoints on the system prompt and the newest message."""
    out = [dict(m) for m in messages]
    for i in (0, len(out) - 1):
        m = out[i]
        if isinstance(m.get("content"), str) and m["content"] and m["role"] in ("system", "user", "tool"):
            m["content"] = [{"type": "text", "text": m["content"], "cache_control": {"type": "ephemeral"}}]
    return out


def _clean(msg: dict) -> dict:
    m = {"role": "assistant", "content": msg.get("content") or ""}
    if msg.get("tool_calls"):
        m["tool_calls"] = msg["tool_calls"]
    return m


async def handback(rw_dir: Path):
    """Make the rw tree owned by the host user again (the :U mount hands it to a subuid of the container's userns)."""
    code, out = await run(["podman", "unshare", "chown", "-R", "0:0", str(rw_dir)], 120)
    if code:
        raise RuntimeError(f"could not restore ownership of {rw_dir}: {out[-300:]}")


async def _submit(box: Box, child: Path, sample_records, locked: dict | None, kind: str) -> list[str]:
    await box.sh("chmod -R a+rX /rw", 30, user="root")  # host-side check must be able to read everything
    errs = []
    if locked is not None:
        try:
            info = json.loads((child / "node.json").read_text())
        except (OSError, json.JSONDecodeError):
            info = {}
        for k in EXEMPT_KEYS:
            if isinstance(info, dict) and info.get(k) != locked.get(k):
                errs.append(f"node.json: {k!r} may not be changed (was {locked.get(k)!r})")
    hyp = child / "hypothesis.md"
    if not hyp.exists() or not hyp.read_text().strip():
        errs.append("hypothesis.md is empty: write one paragraph on why the REAL edit should work and the NULL not")
    errs += await build.check_async(child, sample_records)
    return errs


async def run_agent(
    kind: str,
    ro_dir,
    rw_dir,
    *,
    model=llm.SONNET,
    max_turns=60,
    sample_records=None,
    locked: dict | None = None,
    max_cost: float | None = None,
    reasoning="medium",
    image="evalcue-sandbox",
    log=None,
) -> dict:
    """kind: "evolver" | "creator". locked: node.json as handed to the agent (exemption flags must not change).
    max_cost: stop (unsubmitted) once spend passes this. log: optional path, transcript rewritten each turn."""
    assert kind in ("evolver", "creator")
    ro_dir, rw_dir = Path(ro_dir).resolve(), Path(rw_dir).resolve()
    child = rw_dir / "child"
    messages = [{"role": "system", "content": system_prompt(kind)}, {"role": "user", "content": KICKOFF[kind]}]
    cost, turns, submitted, errors, nudges = 0.0, 0, False, None, 0
    usage_log = []
    t0 = time.time()

    def save():
        if log:
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            Path(log).write_text(
                json.dumps(
                    {
                        "kind": kind,
                        "model": model,
                        "submitted": submitted,
                        "turns": turns,
                        "cost": cost,
                        "errors": errors,
                        "seconds": round(time.time() - t0),
                        "usage": usage_log,
                        "transcript": messages,
                    },
                    indent=1,
                )
            )

    try:
        async with Box({}, cwd="/scratch", image=image, mounts=[(str(ro_dir), "/ro", "ro"), (str(rw_dir), "/rw", "rw")]) as box:
            code, out = await box.sh("chown -R 1000:1000 /rw", 60, user="root")
            if code:
                raise RuntimeError(f"rw setup failed: {out}")
            while turns < max_turns:
                turns += 1
                msg, usage = await llm.ask(_cached(messages), model=model, tools=TOOLS, reasoning=reasoning, max_tokens=16000)
                cost += usage.get("cost") or 0.0
                usage_log.append(usage)
                messages.append({**_clean(msg), "reasoning": msg.get("reasoning") or ""})
                calls = msg.get("tool_calls") or []
                if not calls:
                    if nudges >= 2:
                        break
                    nudges += 1
                    messages.append(
                        {
                            "role": "user",
                            "content": "You have not submitted yet. Keep working with the bash "
                            "tool, and call submit when the node in /rw/child is ready.",
                        }
                    )
                    save()
                    continue
                for c in calls:
                    name = c["function"]["name"]
                    try:
                        args = json.loads(c["function"]["arguments"] or "{}")
                    except json.JSONDecodeError:
                        args = None
                    if name == "bash" and isinstance(args, dict) and "command" in args:
                        code, out = await box.sh(f"cd /rw/child 2>/dev/null; {args['command']}", CMD_TIMEOUT, max_out=MAX_OUT)
                        res = (out if out.strip() else "(no output)") + f"\n[exit code {code}]"
                    elif name == "submit":
                        errors = await _submit(box, child, sample_records, locked, kind)
                        if not errors:
                            submitted = True
                            res = "OK, submitted."
                        else:
                            res = "Build errors (fix them and call submit again):\n" + "\n".join(f"- {e}" for e in errors)
                    else:
                        res = f'error: unknown tool {name!r} or bad arguments (bash takes {{"command": ...}})'
                    left = max_turns - turns
                    if not submitted and left <= 10:
                        res += f"\n[{left} turns left: submit soon]"
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": res})
                save()
                if submitted or (max_cost is not None and cost >= max_cost):
                    break
    finally:
        await handback(rw_dir)
        for p in child.rglob("__pycache__"):
            shutil.rmtree(p, ignore_errors=True)
        save()
    return {"submitted": submitted, "transcript": messages, "cost": cost, "turns": turns, "errors": errors}


async def run_workspace(ws, **kw) -> dict:
    """run_agent on a workspace.Workspace (mutation or creation)."""
    return await run_agent(ws.kind, ws.ro, ws.rw, sample_records=ws.sample_records, locked=ws.child_info, **kw)
