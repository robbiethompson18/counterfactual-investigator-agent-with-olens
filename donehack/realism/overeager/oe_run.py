"""Run over-eagerness episodes: task × variant × item × seed, GLM 5.3 via OpenRouter, podman sandbox.

  uv run python realism/overeager/oe_run.py --tasks payback,calendar --variants base,core --items 16 --seeds 1 --out smoke
Episodes land in realism/runs/overeager/<out>/<task>/<variant>/<item>__<seed>.json (record, messages, env, instance).
"""

import argparse
import asyncio
import datetime as dt
import importlib
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
# isort: off
import realvariants as V  # registers realrun (realism/run.py); must come first
import realrun as R
from harness_variants import VARIANTS, assemble
# isort: on

TASKS = ["groupchat", "payback", "payback_hint", "calendar", "dana", "gitbranch", "impossiblebench", "evilgenie", "rhb_logs", "rhb_bench"]


def load(task):
    """tasks/<task>.py's TASK (loaded by path: donehack/ has its own `tasks` module)."""
    name = f"overeager_task_{task}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "tasks" / f"{task}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name].TASK


def final_text(messages):
    """The visible text the user would read at the end: assistant text from the last turn with a tool call onward."""
    last = max((k for k, m in enumerate(messages) if m["role"] == "assistant" and m.get("tool_calls")), default=0)
    return "\n".join(m.get("content") or "" for m in messages[last:] if m["role"] == "assistant").strip()


def build_spec(task, variant, i, seed=0, today=None):
    T = load(task)
    inst = T.instance(i)
    env = assemble(T, inst, variant, today)
    S = V.base_state({"platform": "impossiblebench", "task_id": f"{task}_{i}", "files": {}, "entry_point": task}, seed)
    S.files, S.desc_path = env["files"], None
    S.sol_path, S.test_path = getattr(T, "sol_path", None), getattr(T, "test_path", None)
    S.system, S.user, S.tools, S.handlers, S.workdir = env["system"], env["user"], env["tools"], env["handlers"], env["workdir"]
    S.setup = T.setup(inst, env["workdir"]) if hasattr(T, "setup") else []
    S.hostname = getattr(T, "hostname", "sandbox")
    S.extra_tmpfs = list(getattr(T, "extra_tmpfs", []))
    S.meta = {"task": task, "variant": variant, "item": i, "atoms": env["atoms"]}
    spec = V.to_spec(S)
    spec.env = dict(getattr(T, "env", {}))
    return spec, T, inst


async def episode(client, key, task, variant, i, seed, outdir):
    out = outdir / task / variant / f"{i}__{seed}.json"
    if out.exists():
        return json.loads(out.read_text())["record"]
    spec, T, inst = build_spec(task, variant, i, seed)
    box = R.Box(re.sub(r"[^a-zA-Z0-9_.-]", "_", f"oe_{task}_{variant}_{i}_{seed}_{random.randrange(1 << 30)}"), spec)
    st = {"actions": [], "calls": []}
    messages = [{"role": "system", "content": spec.system}, {"role": "user", "content": spec.user}]
    end, cost = "max_steps", 0.0
    try:
        await box.start()
        for _ in range(40):
            j = await R.chat(client, "z-ai/glm-5.3", messages, spec.tools, key, reasoning={"effort": "medium"}, max_tokens=None)
            cost += (j.get("usage") or {}).get("cost", 0) or 0
            msg = {k: v for k, v in j["choices"][0]["message"].items() if v is not None and k != "refusal"}
            messages.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                end = "responded"
                break
            stop = False
            for tc in calls:
                nm = tc["function"]["name"]
                try:
                    a = json.loads(tc["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    a = {}
                st["calls"].append({"name": nm, "args": a})
                kind = spec.handlers.get(nm)
                if kind == "done":
                    res, stop = "Task marked complete.", True
                elif kind:
                    res, _ = await R.handle(box, kind, a)
                else:
                    res = T.handle(st, inst, nm, a)
                    if res is None:
                        res = f"Error: unknown tool {nm}"
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": res})
            if stop:
                end = "done"
                break
            if cost > 0.10:
                end = "cut_long"
                break
        if hasattr(T, "inspect"):  # read the sandbox's final state (e.g. git refs) before it goes
            await T.inspect(box, st, inst, spec.workdir)
    finally:
        await box.stop()
    fin = final_text(messages)
    g = T.grade(st, inst)
    if end == "cut_long":  # hit the $0.10 cap mid-episode: unscorable, don't count it either way
        g = {**g, "outcome": "cut_long", "acted": False}
    rec = {
        "task": task,
        "variant": variant,
        "item": i,
        "seed": seed,
        "end": end,
        "cost": cost,
        **g,
        "asked": not g["acted"] and "?" in fin,
        "final_text": fin,
        "actions": st["actions"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    env = {"system": spec.system, "user": spec.user, "tools": spec.tools, "files": spec.files, "workdir": spec.workdir, "setup": spec.setup}
    small = {k: v for k, v in inst.items() if len(json.dumps(v, default=str)) < 200_000}  # drop bulky data (logs, problem files)
    env["files"] = {k: (v if isinstance(v, str) and len(v) < 200_000 else f"<{len(v)} bytes>") for k, v in env["files"].items()}
    blob = {"record": rec, "messages": messages, "env": env, "instance": small}
    out.write_text(json.dumps(blob, indent=1, ensure_ascii=False, default=str))
    return rec


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--items", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--out", default="dev")
    ap.add_argument("--conc", type=int, default=24)
    a = ap.parse_args()
    outdir = HERE.parent / "runs" / "overeager" / a.out
    key = Path("/secrets/openrouter").read_text().strip()
    sem = asyncio.Semaphore(a.conc)
    jobs = [(t, v, i, s) for s in range(a.seeds) for t in a.tasks.split(",") for v in a.variants.split(",") for i in range(a.items)]
    print(f"{len(jobs)} episodes, today={dt.date.today()}", flush=True)

    async def go(j):
        async with sem:
            try:
                r = await episode(client, key, *j, outdir)
                print(j, r["outcome"], "asked" if r["asked"] else "", f"${r['cost']:.4f}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(j, "FAILED", repr(e)[:300], flush=True)

    async with httpx.AsyncClient(timeout=600) as client:
        await asyncio.gather(*[go(j) for j in jobs])


if __name__ == "__main__":
    asyncio.run(main())
