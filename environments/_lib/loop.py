"""Running one episode: the default tool loop (bash + app tools + stop rules). An env with a richer loop defines its own
`step(ep, ctx, call)` coroutine and the runner uses that instead.

Stop rules: a reply without tool calls ends the episode ("responded"); a `done` handler ends it ("done"); `max_turns`
model calls ("max_turns"); a prompt over `max_prompt_tokens` ("cut_long", the realism runs' cap, in tokens not dollars).
Every model call is logged with its provider and usage, including calls whose turn a custom step later drops.
"""

import json
import time
from dataclasses import asdict

import httpx

from .backends import Target, chat
from .box import Box, handle
from .core import Ctx, Episode


async def run_episode(env, ep: Episode, ctx: Ctx, target: Target, client: httpx.AsyncClient, caps: dict) -> dict:
    t0 = time.time()
    messages = [dict(m) for m in ep.messages]
    state = json.loads(json.dumps(ep.state, default=str))
    calls: list[dict] = []
    finish, box, files_after, omitted = "max_turns", None, {}, []

    async def call():
        r = await chat(client, target, messages, ep.tools or None)
        calls.append({"provider": r["provider"], "finish": r["finish"], "usage": r["usage"]})
        return r

    try:
        if ep.needs_box:
            box = Box(ep, f"{ctx.env}_{ctx.arm}_{ctx.item}_{ctx.sample}")
            await box.start()
            for call_id, cmd in ep.prefill:  # commands run for real, shown as the agent's first tool turns
                out = await handle(box, "shell", {"command": cmd})
                tc = {"id": call_id, "type": "function", "function": {"name": ep.prefill_tool, "arguments": json.dumps({"command": cmd})}}
                messages += [
                    {"role": "assistant", "content": "", "tool_calls": [tc]},
                    {"role": "tool", "tool_call_id": call_id, "content": out},
                ]
        if hasattr(env, "step"):
            finish = await env.step(ep, ctx, messages, state, box, call)
        else:
            for _ in range(caps.get("max_turns") or 40):
                r = await call()
                msg = r["message"]
                messages.append(msg)
                if r["finish"] == "length":
                    finish = "length"
                    break
                tcs = msg.get("tool_calls") or []
                if not tcs:
                    finish = "responded"
                    break
                stop = False
                for tc in tcs:
                    nm = tc["function"]["name"]
                    try:
                        a = json.loads(tc["function"].get("arguments") or "{}")
                    except json.JSONDecodeError:
                        a = {}
                    state.setdefault("calls", []).append({"name": nm, "args": a})
                    kind = ep.handlers.get(nm)
                    if kind == "done":
                        res, stop = "Task marked complete.", True
                    elif kind and box:
                        res = await handle(box, kind, a)
                    else:
                        res = env.handle(state, ep.aliases.get(nm, nm), a, ctx) if hasattr(env, "handle") else None
                        if res is None:
                            res = f"Error: unknown tool {nm}"
                    if ep.tool_footer:
                        res += ep.tool_footer.format(n=len(state["calls"]))
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": res})
                if stop:
                    finish = "done"
                    break
                cap = caps.get("max_prompt_tokens")
                if cap and (r["usage"].get("prompt_tokens") or 0) > cap:
                    finish = "cut_long"
                    break
        if box:
            await box.lockdown()
            if hasattr(env, "finalize"):
                await env.finalize(box, state, ctx)
            files_after, omitted = await box.snapshot()
    finally:
        if box:
            await box.stop()
    usage = {
        "prompt_tokens": sum(c["usage"].get("prompt_tokens") or 0 for c in calls),
        "completion_tokens": sum(c["usage"].get("completion_tokens") or 0 for c in calls),
        "cost": sum(c["usage"].get("cost") or 0 for c in calls),
    }
    return {
        "messages": messages,
        "tools": ep.tools,
        "files_before": ep.files,
        "files_after": files_after,
        "omitted": omitted,
        "state": state,
        "meta": ep.meta,
        "setup": ep.setup,
        "workdir": ep.workdir if ep.needs_box else None,
        "finish": finish,
        "turns": len(calls),
        "usage": usage,
        "calls": calls,
        "seconds": round(time.time() - t0, 1),
        "episode_spec": {
            k: v
            for k, v in asdict(ep).items()
            if k in ("image", "hostname", "env", "extra_tmpfs", "prefill", "handlers", "aliases", "tool_footer")
        },
    }
