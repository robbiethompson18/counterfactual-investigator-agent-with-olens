"""The one OpenRouter chat-completions client. Every non-Qwen model call goes through here, not a hand-rolled URL.

    from counterfactual_investigator_agent_with_olens import openrouter
    r = await openrouter.chat("google/gemini-3.8-flash", messages, max_tokens=16000, tools=schemas)
    r.choices[0].message.content, r.usage.cost        # an openai ChatCompletion; r.model_dump() for the raw JSON dict

`chat` sends reasoning effort `medium` unless told otherwise (CLAUDE.md: every non-Qwen model runs at medium; provider
defaults differ), asks for usage accounting (so `r.usage.cost` is the dollar cost) and retries transient errors with
backoff. Anything else (`max_tokens`, `tools`, `temperature`, `seed`, `response_format`, ...) passes through to the SDK.
`client` and `extra_body` are the pieces, for a caller that runs its own loop (evalcue/harness.py, which also serves Qwen).
"""

import asyncio
import os
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, InternalServerError, RateLimitError
from openai.types.chat import ChatCompletion

BASE_URL = "https://openrouter.ai/api/v1"
EFFORT = "medium"
_CLIENT: AsyncOpenAI | None = None


class NoChoices(Exception):
    """OpenRouter reports some provider errors as a 200 with no choices."""


def api_key() -> str:
    """$OPENROUTER_API_KEY, else /secrets/openrouter (bramble)."""
    return os.environ.get("OPENROUTER_API_KEY") or Path("/secrets/openrouter").read_text().strip()


def client(timeout: float = 600, max_retries: int = 2) -> AsyncOpenAI:
    """A fresh AsyncOpenAI pointed at OpenRouter. `chat` keeps one of its own; build one here only when you need the client
    object itself (e.g. raw `completions.create`) or a non-default timeout on every call."""
    return AsyncOpenAI(base_url=BASE_URL, api_key=api_key(), timeout=timeout, max_retries=max_retries)


def extra_body(*, thinking: bool = True, effort: str | None = EFFORT, provider: dict | None = None) -> dict:
    """OpenRouter's own request fields: usage accounting, the unified reasoning switch (`enabled` plus an effort when
    thinking is on; an empty effort leaves the provider's default level) and an optional provider pin, e.g.
    {"order": ["DeepInfra"], "allow_fallbacks": False}."""
    body: dict[str, Any] = {
        "usage": {"include": True},
        "reasoning": {"enabled": thinking, **({"effort": effort} if thinking and effort else {})},
    }
    if provider:
        body["provider"] = provider
    return body


async def chat(
    model: str,
    messages: list[Any],  # plain dicts, as the SDK accepts them
    *,
    effort: str | None = EFFORT,
    provider: dict | None = None,
    timeout: float = 600,
    attempts: int = 6,
    **params: Any,
) -> ChatCompletion:
    """One chat completion with thinking on at `effort`. Connection errors, 5xx, 429 and empty-choices responses are retried
    `attempts` times with 10 s, 20 s, 40 s, ... between tries (~5 min in all for 6, which outlasts OpenRouter's per-minute
    429s); a timeout is not retried, it already waited `timeout` seconds. Other request fields go in `params`; an
    `extra_body` there is merged over the OpenRouter fields."""
    global _CLIENT
    if _CLIENT is None:  # built on first use, not at import: modules get imported on machines without the key
        _CLIENT = client()
    extra = extra_body(effort=effort, provider=provider) | (params.pop("extra_body", None) or {})
    for attempt in range(attempts):
        try:
            r = await _CLIENT.chat.completions.create(model=model, messages=messages, timeout=timeout, extra_body=extra, **params)
            if not getattr(r, "choices", None):
                raise NoChoices(str(r.model_extra)[:500])
            return r
        except (APIConnectionError, InternalServerError, RateLimitError, NoChoices) as e:
            if attempt == attempts - 1 or isinstance(e, APITimeoutError):
                raise
            await asyncio.sleep(10 * 2**attempt)
    raise AssertionError("unreachable")  # the loop returns or raises
