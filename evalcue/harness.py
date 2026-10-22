"""Shared harness for multi-arm behavioral environments on Qwen3.6-27B (reward hacking, self-preservation, hiring bias, ...).
See evalcue/README.md.

An env is a module in evalcue/envs/ with:
    ARMS: list[str]                                   all arms are peers; no arm is the baseline
    items() -> list[str]                              task-instance ids (samples are spread round-robin over them)
    build(arm, item) -> Episode                       everything the model sees
    detect(row) -> dict                               {"positive": bool | None, ...}; None = unscorable (excluded, counted)
                                                      (older envs return "hack"; it's read as "positive")
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
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI, omit

# the shared Qwen3.6-27B server ($QWEN_VLLM_URL / $QWEN_VLLM_API_KEY, see runpod/CLAUDE.md); EVALCUE_URLS (comma-separated,
# round-robin) overrides it, e.g. for your own tunnelled pods
URLS = os.environ.get("EVALCUE_URLS", os.environ.get("QWEN_VLLM_URL", "http://127.0.0.1:18000/v1")).split(",")
API_KEY = os.environ.get("QWEN_VLLM_API_KEY", "none")
CLIENTS = [AsyncOpenAI(base_url=u, api_key=API_KEY, timeout=6 * 3600) for u in URLS]  # 100k+ token thinking traces take hours
MODEL = "qwen"
# fallback for comparison only: EVALCUE_MODEL=anthropic/claude-haiku-5.5 (or google/gemini-3.5-flash-lite) goes to OpenRouter
if os.environ.get("EVALCUE_MODEL"):
    MODEL = os.environ["EVALCUE_MODEL"]
    CLIENTS = [AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=6 * 3600)]
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


def _client(i: int) -> AsyncOpenAI:
    return CLIENTS[i % len(CLIENTS)]


_counter = 0


async def chat(messages, *, thinking: bool, max_tokens: int | None, tools=None):
    global _counter
    _counter += 1
    temperature, top_p, presence_penalty = SAMPLING[thinking]
    extra = {"top_k": 20, "chat_template_kwargs": {"enable_thinking": thinking, "preserve_thinking": True}}
    if MODEL != "qwen":  # OpenRouter's unified reasoning switch
        extra = {"reasoning": {"enabled": thinking}}
    r = await _client(_counter).chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens or omit,  # None: up to the server's context limit
        tools=tools or omit,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        extra_body=extra,
    )
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

    def __init__(self, files: dict[str, str], cwd: str, env: dict[str, str]):
        self.dir = Path(tempfile.mkdtemp(prefix="evalcue-"))
        for rel, text in files.items():
            p = self.dir / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            if text.startswith("#!"):  # scripts like ./next_question or send_email can run without `python`
                p.chmod(0o755)
        envs = [a for k, v in env.items() for a in ("-e", f"{k}={v}")]
        self.cid = subprocess.check_output(
            ["docker", "run", "-d", "--rm", "--network", "none", "--memory", "512m", "-v", f"{self.dir}:{cwd}", "-w", cwd, *envs]
            + [SANDBOX_IMAGE, "sleep", "3600"],
            text=True,
        ).strip()

    def run(self, command: str, timeout: int = 30) -> tuple[str, int]:
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
        return (out if len(out) < 8000 else out[:4000] + "\n...[truncated]...\n" + out[-4000:]), code

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


def run_python(code: str, timeout: int = 10, stdin: str = "") -> tuple[str, int]:
    """Run untrusted model code in a throwaway no-network container. Returns (output, exit code)."""
    try:
        p = subprocess.run(
            ["docker", "run", "--rm", "-i", "--network", "none", "--memory", "512m", SANDBOX_IMAGE, "timeout", str(timeout)]
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
    if not ep.files:
        msg, finish, ntok = await chat(msgs, thinking=thinking, max_tokens=max_tokens)
        return {"messages": msgs + [msg], "finish": finish, "turns": 1, "completion_tokens": ntok, "files_after": {}}
    sb = await asyncio.to_thread(Sandbox, ep.files, ep.cwd, ep.env)
    total, finish, turns = 0, "max_turns", 0
    try:
        for turns in range(1, ep.max_turns + 1):
            msg, f, ntok = await chat(msgs, thinking=thinking, max_tokens=max_tokens, tools=[BASH_TOOL])
            total += ntok
            msgs.append(msg)
            if f == "length":
                finish = "length"
                break
            if not msg.get("tool_calls"):
                finish = "stop"
                break
            for tc in msg["tool_calls"]:
                try:
                    cmd = json.loads(tc["function"]["arguments"]).get("command", "")
                    out, code = await asyncio.to_thread(sb.run, cmd)
                    content = f"{out}\n[exit code {code}]"
                except (json.JSONDecodeError, AttributeError):
                    content = "error: arguments must be JSON with a 'command' string"
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": content})
        files_after = await asyncio.to_thread(sb.snapshot)
    finally:
        await asyncio.to_thread(sb.close)
    return {"messages": msgs, "finish": finish, "turns": turns, "completion_tokens": total, "files_after": files_after}


async def run_batch(env, *, arms, n_per_arm, thinking, max_tokens, out: Path, concurrency=512, seed=0):
    """n_per_arm samples per arm, spread round-robin over env.items(); arms interleaved so endpoint drift can't confound."""
    items = env.items()
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
                "env": env.__name__.split(".")[-1],
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


def score(env, row) -> dict:
    """env.detect, except a multi-turn episode cut off by the token cap with no positive yet is unscorable (it might have
    happened later)."""
    out = env.detect(row)
    if "positive" not in out:  # envs written before the rename return "hack"
        out["positive"] = out.get("hack")
    if row.get("finish") == "length" and row.get("files_before") and not out["positive"]:
        out["positive"] = None
    return out


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open()]
    for r in rows:
        r.setdefault("positive", r.get("hack"))  # rows written before the rename
    return rows


__all__ = ["Episode", "asdict", "last_python_block", "load_rows", "run_batch", "run_python", "summarize"]
