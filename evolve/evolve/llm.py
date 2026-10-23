"""Model clients. Target = Qwen3.6-27B on our vLLM server; DS (forecaster, summariser, auditor) and Sonnet (evolver,
creator) go through OpenRouter. Keys are read from files, never logged."""

import asyncio
import os
import random
from pathlib import Path

from openai import AsyncOpenAI

QWEN_URL = os.environ.get("QWEN_VLLM_URL", "http://127.0.0.1:18000/v1")
QWEN_KEY_FILE = Path(os.environ.get("QWEN_VLLM_KEY_FILE", Path.home() / "workspace/env-explorer/replication/vllm.key"))
OPENROUTER_KEY_FILE = Path(os.environ.get("OPENROUTER_KEY_FILE", "/secrets/openrouter"))

DS = "deepseek/deepseek-v4.1-flash-1017"
DS_PROVIDER = {"order": ["DeepInfra"], "allow_fallbacks": True}
SONNET = "anthropic/claude-sonnet-5"
# Qwen3.6 model-card sampling per mode: (temperature, top_p, presence_penalty), top_k 20 (same as Robbie's harness)
QWEN_SAMPLING = {True: (1.0, 0.95, 0.0), False: (0.7, 0.8, 1.5)}

_qwen = _router = None


def qwen() -> AsyncOpenAI:
    global _qwen
    if _qwen is None:
        _qwen = AsyncOpenAI(base_url=QWEN_URL, api_key=QWEN_KEY_FILE.read_text().strip(), timeout=6 * 3600, max_retries=3)
    return _qwen


def router() -> AsyncOpenAI:
    global _router
    if _router is None:
        _router = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY_FILE.read_text().strip(), timeout=1800, max_retries=2
        )
    return _router


async def target_chat(messages, tools, *, thinking=True, max_tokens=None):
    """One Qwen turn. Returns (assistant message dict, finish_reason, completion tokens)."""
    t, p, pp = QWEN_SAMPLING[thinking]
    kw = {
        "model": "qwen",
        "messages": messages,
        "temperature": t,
        "top_p": p,
        "presence_penalty": pp,
        "extra_body": {"top_k": 20, "chat_template_kwargs": {"enable_thinking": thinking, "preserve_thinking": True}},
    }
    if tools:
        kw["tools"] = tools
    if max_tokens:
        kw["max_tokens"] = max_tokens
    r = await qwen().chat.completions.create(**kw)
    m = r.choices[0].message
    reasoning = getattr(m, "reasoning_content", None) or (m.model_extra or {}).get("reasoning") or ""
    msg = {"role": "assistant", "content": m.content or "", "reasoning_content": reasoning}
    if m.tool_calls:
        msg["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in m.tool_calls
            if c.type == "function"
        ]
    return msg, r.choices[0].finish_reason, (r.usage.completion_tokens if r.usage else 0)


async def ask(messages, *, model=DS, tools=None, reasoning="medium", max_tokens=32000, retries=5):
    """One OpenRouter completion with retries. Returns (message dict, usage dict incl. cost)."""
    body = {"reasoning": {"effort": reasoning}, "usage": {"include": True}}
    if model == DS:
        body["provider"] = DS_PROVIDER
    for attempt in range(retries):
        try:
            kw = {"model": model, "messages": messages, "max_tokens": max_tokens, "extra_body": body}
            if tools:
                kw["tools"] = tools
            r = await router().chat.completions.create(**kw)
            if not r.choices:
                raise RuntimeError(f"no choices: {r}")
            m = r.choices[0].message
            msg = {
                "role": "assistant",
                "content": m.content or "",
                "reasoning": getattr(m, "reasoning", None) or (m.model_extra or {}).get("reasoning") or "",
            }
            if m.tool_calls:
                msg["tool_calls"] = [
                    {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                    for c in m.tool_calls
                ]
            u = r.usage.model_dump() if r.usage else {}
            return msg, {
                "cost": u.get("cost") or 0.0,
                "prompt_tokens": u.get("prompt_tokens"),
                "completion_tokens": u.get("completion_tokens"),
                "finish": r.choices[0].finish_reason,
            }
        except Exception:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(min(60, 2**attempt * 3) * (0.5 + random.random()))
