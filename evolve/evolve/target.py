"""Run the target model (Qwen) on one arm and return a record.

An arm with no tools is a plain chat: one completion. An arm with tools runs in a sandbox: each turn the target may
call tools, implemented as
    {"impl": {"kind": "bash"}}                      run args["command"] in the work dir
    {"impl": {"kind": "script", "path": "bin/x"}}   run a work-dir script with the JSON args on stdin
    {"impl": {"kind": "record", "text": "..."}}     do nothing, return fixed text
and the episode ends at the first stop rule hit ("reply" is always on: a turn with no tool call ends it):
    tool:<name>  command:<regex>  max_turns:<n>  file_exists:<path>
plus "length" when the target runs out of tokens.

The record (what the grader gets) holds only what the target did and saw back, never the setup:
    final_response   text of the last assistant turn
    turns            [{text, reasoning, tool_calls: [{name, args, output}]}], one per assistant turn
    tool_calls       the same tool calls, flattened
    transcript       raw assistant/tool messages after the prompt (OpenAI format)
    files_after      sandbox work dir after the episode (read-only files restored; see readonly_tampered)
    finish           reply | length | max_turns | tool:<name> | command | file_exists:<path>
    meta             the arm's meta.json
"""

import json
import re

from . import llm
from .node import Arm
from .sandbox import Box

TOOL_TIMEOUT = 30


def _openai_tools(tools):
    return [{"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters")}} for t in tools]


def _stops(rules):
    out = {"tool": set(), "command": [], "max_turns": 20, "file_exists": []}
    for r in rules:
        kind, _, arg = r.partition(":")
        if kind == "tool":
            out["tool"].add(arg)
        elif kind == "command":
            out["command"].append(re.compile(arg))
        elif kind == "max_turns":
            out["max_turns"] = int(arg)
        elif kind == "file_exists":
            out["file_exists"].append(arg)
    return out


async def _call(box: Box, tool: dict, args: dict) -> str:
    """Optional impl fields: timeout (s), max_out (chars), format (with {out} and {code})."""
    impl = tool.get("impl", {"kind": "bash"})
    if impl["kind"] == "record":
        return impl.get("text", "OK.")
    t, mo = impl.get("timeout", TOOL_TIMEOUT), impl.get("max_out", 8000)
    if impl["kind"] == "bash":
        code, out = await box.sh(str(args.get("command", "")), t, max_out=mo)
    else:  # script
        code, out = await box.sh(f"./{impl['path']}", t, stdin=json.dumps(args), max_out=mo)
    return impl.get("format", "{out}\n[exit code {code}]").replace("{out}", out).replace("{code}", str(code))


def _args(call):
    try:
        return json.loads(call["function"]["arguments"] or "{}")
    except (json.JSONDecodeError, TypeError):
        return {"_raw": call["function"]["arguments"]}


TOKEN_BUDGET = 32000  # completion tokens per episode (Kylie, 2026-10-23); normal episodes use < 32k, runaway
# thinking loops otherwise run to the ~260k context limit (~100 min). An episode that hits it ends with finish "length".


def _budget(ep: dict, used: int) -> int:
    return min(ep.get("max_tokens") or TOKEN_BUDGET, TOKEN_BUDGET - used)


async def run_episode(arm: Arm) -> dict:
    ep = arm.episode
    prompt = [dict(m) for m in arm.messages]
    n_prompt = len(prompt)
    msgs = list(prompt)
    record = {"tool_calls": [], "n_turns": 0, "finish": "reply", "files_after": {}, "readonly_tampered": [], "meta": arm.meta}
    tok = 0
    if not arm.tools:
        msg, finish, tok = await llm.target_chat(msgs, None, thinking=ep.get("thinking", True), max_tokens=_budget(ep, 0))
        msgs.append(msg)
        record.update(n_turns=1, finish="length" if finish == "length" else "reply")
    else:
        stop = _stops(ep.get("stop", []))
        by_name = {t["name"]: t for t in arm.tools}
        async with Box(
            arm.files,
            cwd=ep.get("cwd", "/workspace"),
            env=ep.get("env"),
            readonly=ep.get("readonly", []),
            image=ep.get("image", "evalcue-sandbox"),
        ) as box:
            record["finish"] = "max_turns"
            for turn in range(1, stop["max_turns"] + 1):
                record["n_turns"] = turn
                msg, finish, n = await llm.target_chat(
                    msgs, _openai_tools(arm.tools), thinking=ep.get("thinking", True), max_tokens=_budget(ep, tok)
                )
                tok += n
                msgs.append(msg)
                if finish == "length" or tok >= TOKEN_BUDGET:
                    record["finish"] = "length"
                    break
                if not msg.get("tool_calls"):
                    record["finish"] = "reply"
                    break
                hit = None
                for tc in msg["tool_calls"]:
                    name = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"] or "{}")
                        out = await _call(box, by_name[name], args) if name in by_name else f"error: no tool named {name}"
                    except (json.JSONDecodeError, TypeError):
                        args, out = {"_raw": tc["function"]["arguments"]}, "error: arguments must be a JSON object"
                    msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
                    record["tool_calls"].append({"name": name, "args": args, "output": out})
                    if name in stop["tool"]:
                        hit = f"tool:{name}"
                    elif (
                        name in by_name
                        and by_name[name].get("impl", {}).get("kind", "bash") == "bash"
                        and any(r.search(str(args.get("command", ""))) for r in stop["command"])
                    ):
                        hit = "command"
                for path in stop["file_exists"]:
                    if not hit and (await box.sh(f"test -e '{path}'", 10))[0] == 0:
                        hit = f"file_exists:{path}"
                if hit:
                    record["finish"] = hit
                    break
            files = await box.snapshot()
        for p in ep.get("readonly", []):  # the grader sees read-only files as they were; tampering is reported, not hidden
            if files.get(p) != arm.files.get(p):
                record["readonly_tampered"].append(p)
            if p in arm.files:
                files[p] = arm.files[p]
        record["files_after"] = files
    record["transcript"] = msgs[n_prompt:]
    outputs = {m["tool_call_id"]: m["content"] for m in record["transcript"] if m["role"] == "tool"}
    record["turns"] = [
        {
            "text": m.get("content") or "",
            "reasoning": m.get("reasoning_content") or "",
            "tool_calls": [
                {"name": c["function"]["name"], "args": _args(c), "output": outputs.get(c["id"], "")} for c in m.get("tool_calls", [])
            ],
        }
        for m in record["transcript"]
        if m["role"] == "assistant"
    ]
    record["final_response"] = next((m.get("content") or "" for m in reversed(record["transcript"]) if m["role"] == "assistant"), "")
    record["completion_tokens"] = tok
    return record
