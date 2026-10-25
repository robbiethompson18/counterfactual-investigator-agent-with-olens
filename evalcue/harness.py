"""Shared harness for multi-arm behavioral environments on Qwen3.6-27B (reward hacking, self-preservation, hiring bias, ...).
See evalcue/README.md.

An env is a module in evalcue/envs/ with:
    ARMS: list[str]                                   all arms are peers; no arm is the baseline
    items() -> list[str]                              task-instance ids (samples are spread round-robin over them)
    build(arm, item) -> Episode                       everything the model sees
    detect(row) -> dict                               {"positive": bool | None, ...}; None = unscorable (excluded, counted)
                                                      (older envs return "hack"; it's read as "positive")
    BEHAVIOR: str                                     optional: detect()'s rule in words, shown to cfbench's predictors
Generation and detection are separate: rows store the full transcript (and the sandbox's final files), so
detect() can change and be re-run offline with `python -m evalcue.run ENV --rescore FILE`.
"""

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, InternalServerError, RateLimitError, omit

from counterfactual_investigator_agent_with_olens import qwen

# the shared Qwen3.6-27B server ($QWEN_VLLM_URL / $QWEN_VLLM_API_KEY, see runpod/CLAUDE.md); recorded in meta.json
URL = os.environ.get("QWEN_VLLM_URL", "")
# qwen.client sets TCP keepalives: without them a long silent generation's connection gets dropped on the way to RunPod and
# the client waits forever on a response that was already sent
MODEL = os.environ.get("EVALCUE_MODEL", "qwen")
if MODEL == "qwen":
    CLIENT = qwen.client()
else:  # fallback for comparison only: EVALCUE_MODEL=anthropic/claude-haiku-5.5 (or google/gemini-3.5-flash-lite) goes to OpenRouter
    CLIENT = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=6 * 3600)
# EVALCUE_PROVIDER pins OpenRouter to one provider slug (no fallbacks), so a run's rates don't mix quantizations/deployments
PROVIDER = os.environ.get("EVALCUE_PROVIDER")
# OpenRouter reasoning effort when thinking is on; medium everywhere by default (provider defaults vary: GLM 5.3's is max)
EFFORT = os.environ.get("EVALCUE_EFFORT", "medium")
SANDBOX_IMAGE = "evalcue-sandbox"  # built from evalcue/Dockerfile: python 3.12 + pytest + numpy + git
# Qwen3.6 model-card sampling defaults per mode
SAMPLING = {True: (1.0, 0.95, 0.0), False: (0.7, 0.8, 1.5)}  # (temperature, top_p, presence_penalty), top_k 20 for both

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a bash command in the working directory. Returns stdout+stderr (truncated) and the exit code.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    },
}


@dataclass
class Episode:
    messages: list[dict]  # system/user (and optionally prefilled assistant/tool) turns
    files: dict[str, str] = field(default_factory=dict)  # sandbox files, relative to cwd; non-empty => multi-turn bash agent
    cwd: str = "/workspace"  # where the files are mounted in the sandbox; the cue can live here
    env: dict[str, str] = field(default_factory=dict)  # sandbox env vars
    max_turns: int = 20
    meta: dict = field(default_factory=dict)  # anything detect() needs (gold answer, mutated test, cue text, ...)
    image: str = SANDBOX_IMAGE
    budget: int | None = None  # max commands: each tool output says how many are left; the episode ends once they're spent...
    final_command: str | None = None  # ...unless this regex is set: then only matching commands run, 3 tries (e.g. submit)
    out_of_budget: str = ""  # appended to the output that spends the budget, and to each refused command (e.g. "submit now")
    done_file: str | None = None  # the episode ends as soon as this file exists in the sandbox (e.g. written by a submit script)
    tools: list[dict] = field(default_factory=lambda: [BASH_TOOL])
    # Replaces the bash handling of each assistant turn, for ports that reproduce another harness's tool loop (its tool
    # names, output format, retry messages): (sandbox, assistant message) -> (messages to append, finish reason or None).
    step: Callable[["Sandbox | None", dict], tuple[list[dict], str | None]] | None = None  # sandbox is None without files
    hostname: str | None = None  # the container's hostname (docker run --hostname); None keeps docker's random id


async def chat(messages, *, thinking: bool, max_tokens: int | None, tools=None):
    temperature, top_p, presence_penalty = SAMPLING[thinking]
    extra = {"top_k": 20, "chat_template_kwargs": {"enable_thinking": thinking, "preserve_thinking": True}}
    if MODEL != "qwen":  # OpenRouter's unified reasoning switch
        extra = {"reasoning": {"enabled": thinking, **({"effort": EFFORT} if thinking and EFFORT else {})}}
        if PROVIDER:
            extra["provider"] = {"order": [PROVIDER], "allow_fallbacks": False}
    # Transient errors get retried with backoff (~5 min in all) instead of becoming errored rows: drops hit long
    # generations most, so excluding them would bias every arm's rate toward short episodes. A longer outage still errors.
    for attempt in range(6):
        try:
            r = await CLIENT.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=max_tokens or omit,  # None: up to the server's context limit
                tools=tools or omit,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                extra_body=extra,
            )
            break
        except (APIConnectionError, InternalServerError, RateLimitError) as e:
            if attempt == 5 or isinstance(e, APITimeoutError):  # a timeout already waited the client's full 6 h
                raise
            await asyncio.sleep(10 * 2**attempt)
    m = r.choices[0].message
    reasoning = getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None) or (m.model_extra or {}).get("reasoning")
    msg = {"role": "assistant", "content": m.content or "", "reasoning_content": reasoning or ""}
    if m.tool_calls:
        msg["tool_calls"] = [
            {"id": t.id, "type": "function", "function": {"name": t.function.name, "arguments": t.function.arguments}}
            for t in m.tool_calls
            if t.type == "function"
        ]
    return msg, r.choices[0].finish_reason, r.usage.completion_tokens if r.usage else 0


# ---------- sandbox (Docker, no network) ----------


class Sandbox:
    """One container per episode, episode files bind-mounted at `cwd` so `pwd` and paths show the cue."""

    def __init__(self, files: dict[str, str], cwd: str, env: dict[str, str], image: str = SANDBOX_IMAGE, hostname: str | None = None):
        self.dir = Path(tempfile.mkdtemp())  # default "tmp" prefix: the host path shows in the container's /proc/self/mountinfo
        for rel, text in files.items():
            p = self.dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            if text.startswith("#!"):  # scripts like ./next_question or send_email can run without `python`
                p.chmod(0o755)
        envs = [a for k, v in env.items() for a in ("-e", f"{k}={v}")]
        envs += ["--hostname", hostname] if hostname else []
        self.cid = subprocess.check_output(
            ["docker", "run", "-d", "--rm", "--network", "none", "--memory", "512m", "-v", f"{self.dir}:{cwd}", "-w", cwd, *envs]
            + [image, "sleep", "3600"],
            text=True,
        ).strip()

    def run(self, command: str, timeout: int = 30, limit: int = 8000) -> tuple[str, int]:
        try:
            p = subprocess.run(
                ["docker", "exec", self.cid, "timeout", str(timeout), "bash", "-lc", command],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout + 10,
            )
            out, code = p.stdout + p.stderr, p.returncode
        except subprocess.TimeoutExpired:
            out, code = "[timed out]", 124
        return (out if len(out) < limit else out[: limit // 2] + "\n...[truncated]...\n" + out[-limit // 2 :]), code

    def snapshot(self) -> dict[str, str]:
        files = {}
        for p in self.dir.rglob("*"):
            if p.is_file() and not {".git", ".pytest_cache", "__pycache__"} & set(p.parts) and p.stat().st_size < 200_000:
                try:
                    files[str(p.relative_to(self.dir))] = p.read_text()
                except UnicodeDecodeError:
                    pass
        return files

    def close(self):
        subprocess.run(["docker", "rm", "-f", self.cid], check=False, capture_output=True)
        shutil.rmtree(self.dir, ignore_errors=True)


def run_python(code: str, timeout: int = 10, stdin: str = "", image: str = SANDBOX_IMAGE) -> tuple[str, int]:
    """Run untrusted model code in a throwaway no-network container. Returns (output, exit code)."""
    try:
        p = subprocess.run(
            ["docker", "run", "--rm", "-i", "--network", "none", "--memory", "512m", image, "timeout", str(timeout)]
            + ["bash", "-c", "cat > /tmp/m.py && python -I /tmp/m.py < /dev/null"],
            check=False,
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout + 20,
        )
        return p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired:
        return "[timed out]", 124


def last_python_block(text: str) -> str | None:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return blocks[-1] if blocks else None


# ---------- running ----------


async def run_episode(ep: Episode, *, thinking: bool, max_tokens: int | None) -> dict:
    msgs = [dict(m) for m in ep.messages]
    if not ep.files and not ep.step:
        msg, finish, ntok = await chat(msgs, thinking=thinking, max_tokens=max_tokens)
        return {"messages": msgs + [msg], "finish": finish, "turns": 1, "completion_tokens": ntok, "files_after": {}}
    # a `step` env with no files keeps its state in Python (e.g. injection_slack's fake Slack), so it needs no container
    sb = await asyncio.to_thread(Sandbox, ep.files, ep.cwd, ep.env, ep.image, ep.hostname) if ep.files else None
    total, finish, turns, used, refused = 0, "max_turns", 0, 0, 0
    try:
        for turns in range(1, ep.max_turns + 1):
            msg, f, ntok = await chat(msgs, thinking=thinking, max_tokens=max_tokens, tools=ep.tools)
            total += ntok
            msgs.append(msg)
            if f == "length":
                finish = "length"
                break
            if ep.step:
                new, end = await asyncio.to_thread(ep.step, sb, msg)
                msgs += new
                if end:
                    finish = end
                    break
                continue
            if not msg.get("tool_calls"):
                finish = "stop"
                break
            assert sb is not None  # only step envs run without files
            for tc in msg["tool_calls"]:
                try:
                    cmd = str(json.loads(tc["function"]["arguments"]).get("command", ""))
                except (json.JSONDecodeError, AttributeError):
                    cmd = None
                spent = ep.budget is not None and used >= ep.budget
                if spent and (refused >= 3 or not ep.final_command or not re.search(ep.final_command, cmd or "")):
                    refused += 1
                    content = f"error: no commands left; this command was not run. {ep.out_of_budget}"
                elif cmd is None:
                    content = "error: arguments must be JSON with a 'command' string"
                else:
                    out, code = await asyncio.to_thread(sb.run, cmd)
                    content = f"{out}\n[exit code {code}]"
                    if spent:  # a final command ran; if it didn't finish the episode, it counts as a try
                        refused += 1
                        content += "\n" + ep.out_of_budget
                    else:
                        used += 1
                        if ep.budget is not None:
                            content += f"\n[{ep.budget - used} of {ep.budget} commands left]"
                            if used == ep.budget:
                                content += "\n" + ep.out_of_budget
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": content})
            if ep.done_file and (sb.dir / ep.done_file).exists():
                finish = "done"
                break
            if ep.budget is not None and used >= ep.budget and (not ep.final_command or refused >= 3):
                finish = "budget"
                break
        files_after = await asyncio.to_thread(sb.snapshot) if sb else {}
    finally:
        if sb:
            await asyncio.to_thread(sb.close)
    return {"messages": msgs, "finish": finish, "turns": turns, "completion_tokens": total, "files_after": files_after}


async def run_batch(env, *, arms, n_per_arm, thinking, max_tokens, out: Path, concurrency=512, seed=0):
    """n_per_arm samples per arm, spread round-robin over env.items(); arms interleaved so endpoint drift can't confound."""
    items = env.items()
    assert items, f"{env.__name__}.items() returned no items"
    rng = np.random.default_rng(seed)
    order = [items[(i + seed) % len(items)] for i in range(n_per_arm)]
    jobs = [(arm, item, i) for i, item in enumerate(order) for arm in arms]
    rng.shuffle(jobs)
    sem = asyncio.Semaphore(concurrency)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = 0

    async def one(arm, item, i):
        nonlocal done
        async with sem:
            ep = env.build(arm, item)
            try:
                res = await run_episode(ep, thinking=thinking, max_tokens=max_tokens)
            except Exception as e:  # noqa: BLE001 - log and keep the batch going
                res = {"error": repr(e)[:500]}
            row = {
                "env": env.__name__.removeprefix("evalcue.envs.").replace(".", "_"),
                "model": MODEL,
                "arm": arm,
                "item": item,
                "sample": i,
                "seed": seed,
                "thinking": thinking,
                **res,
            }
            row["meta"] = ep.meta
            row["files_before"] = ep.files
            row["cwd"] = ep.cwd
            if "error" not in res:
                row.update(await asyncio.to_thread(score, env, row))
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)

    await asyncio.gather(*(one(*j) for j in jobs))


# ---------- stats ----------


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(rows, arms):
    """Every arm's rate. Arms are symmetric; compare whichever pairs are interesting yourself."""
    arms = [a for a in arms if any(r["arm"] == a for r in rows)]  # e.g. --rescore of a file that ran a subset of env.ARMS
    scored = [r for r in rows if r.get("positive") is not None]
    print(f"rows {len(rows)}  scored {len(scored)}  errors {sum('error' in r for r in rows)}  unscorable {len(rows) - len(scored)}")
    for arm in arms:
        ar = [r for r in rows if r["arm"] == arm]
        sc = [r for r in ar if r.get("positive") is not None]
        k = sum(r["positive"] for r in sc)
        lo, hi = wilson(k, len(sc))
        trunc = sum(r.get("finish") in ("length", "max_turns") for r in ar)
        rate = f"{k:3d}/{len(sc):3d} = {k / max(len(sc), 1):5.1%}  [{lo:.0%}, {hi:.0%}]"
        print(f"  {arm:>24}: {rate}  unscorable {len(ar) - len(sc)}  truncated {trunc}")


# Keys a row has before scoring. score() adds only `positive` and `detect_notes`; rows scored before 2026-10-22 had
# detect()'s other fields at the top level, which --rescore strips.
ROW_KEYS = {"env", "model", "arm", "item", "sample", "seed", "thinking", "messages", "finish", "turns", "completion_tokens"}
ROW_KEYS |= {"files_before", "files_after", "meta", "cwd", "error"}


def score(env, row) -> dict:
    """{"positive": the label, "detect_notes": everything else env.detect returned}. A multi-turn episode cut off by the
    token cap with no positive yet is unscorable (it might have happened later)."""
    notes = env.detect(row)
    positive = notes.pop("positive") if "positive" in notes else notes.pop("hack", None)  # older envs return "hack"
    if row.get("finish") == "length" and (row.get("files_before") or row.get("turns", 1) > 1) and not positive:
        positive = None
    return {"positive": positive, "detect_notes": notes}


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open()]
    for r in rows:
        r.setdefault("positive", r.get("hack"))  # rows written before the rename
    return rows


__all__ = ["Episode", "asdict", "last_python_block", "load_rows", "run_batch", "run_python", "summarize"]
