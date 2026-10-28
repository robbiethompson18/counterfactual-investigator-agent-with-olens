"""The one OpenRouter chat-completions client. Every non-Qwen model call goes through here, not a hand-rolled URL.

    from counterfactual_investigator_agent_with_olens import openrouter
    r = await openrouter.chat("google/gemini-3.8-flash", messages, max_tokens=16000, tools=schemas)
    r.choices[0].message.content, r.usage.cost        # an openai ChatCompletion; r.model_dump() for the raw JSON dict

`chat` sends reasoning effort `medium` unless told otherwise (CLAUDE.md: every non-Qwen model runs at medium; provider
defaults differ; `effort="off"` disables thinking, which some models refuse with a 400), asks for usage accounting (so
`r.usage.cost` is the dollar cost) and retries transient errors with backoff. Anything else (`max_tokens`, `tools`,
`temperature`, `seed`, `response_format`, ...) passes through to the SDK.
`client` and `extra_body` are the pieces, for a caller that runs its own loop (evalcue/harness.py, which also serves Qwen).

`gemini-direct/<model>` (e.g. `gemini-direct/gemini-3.8-flash`) skips OpenRouter: the same call goes to Google's
OpenAI-compatible API with $GEMINI_API_KEY, at `reasoning_effort` = effort, so a run can use a key whose daily quota isn't
the one OpenRouter's BYOK route is spending (set GEMINI_API_KEY to the key you want for that run). No `usage.cost`.
`vertex/<model>` does the same on Vertex AI's OpenAI-compatible endpoint in $VERTEX_PROJECT, authenticated as the active
gcloud account (token refreshed every 45 min); for when AI Studio 503s "high demand" but Vertex has capacity.
"""

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, InternalServerError, RateLimitError
from openai.types.chat import ChatCompletion

BASE_URL = "https://openrouter.ai/api/v1"
GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DIRECT = "gemini-direct/"
VERTEX = "vertex/"
EFFORT = "medium"
_CLIENT: AsyncOpenAI | None = None
_GOOGLE: AsyncOpenAI | None = None
_VERTEX: tuple[float, AsyncOpenAI] | None = None  # (built at, client): gcloud access tokens last an hour


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
    """One chat completion with thinking on at `effort`. Connection errors, 5xx, 429, empty-choices and non-JSON responses are retried
    `attempts` times with 10 s, 20 s, 40 s, ... between tries (~5 min in all for 6, which outlasts OpenRouter's per-minute
    429s); a timeout is not retried, it already waited `timeout` seconds. Other request fields go in `params`; an
    `extra_body` there is merged over the OpenRouter fields."""
    global _CLIENT, _GOOGLE, _VERTEX
    if model.startswith(VERTEX):
        if _VERTEX is None or time.time() - _VERTEX[0] > 45 * 60:
            cmd = ["gcloud", "auth", "print-access-token"]
            token = str(await asyncio.to_thread(subprocess.check_output, cmd, text=True)).strip()
            url = f"https://aiplatform.googleapis.com/v1/projects/{os.environ['VERTEX_PROJECT']}/locations/global/endpoints/openapi"
            _VERTEX = (time.time(), AsyncOpenAI(base_url=url, api_key=token, timeout=timeout, max_retries=2))
        api, model = _VERTEX[1], "google/" + model.removeprefix(VERTEX)
        extra = params.pop("extra_body", None) or {}
        params["reasoning_effort"] = "none" if effort == "off" else effort
    elif model.startswith(DIRECT):
        if _GOOGLE is None:
            _GOOGLE = AsyncOpenAI(base_url=GOOGLE_URL, api_key=os.environ["GEMINI_API_KEY"], timeout=timeout, max_retries=2)
        api, model = _GOOGLE, model.removeprefix(DIRECT)
        extra = params.pop("extra_body", None) or {}
        params["reasoning_effort"] = "none" if effort == "off" else effort
    else:
        if _CLIENT is None:  # built on first use, not at import: modules get imported on machines without the key
            _CLIENT = client()
        api = _CLIENT
        extra = extra_body(thinking=effort != "off", effort=effort, provider=provider) | (params.pop("extra_body", None) or {})
    for attempt in range(attempts):
        try:
            r = await api.chat.completions.create(model=model, messages=messages, timeout=timeout, extra_body=extra, **params)
            if not getattr(r, "choices", None):
                raise NoChoices(str(r.model_extra)[:500])
            return r
        except (APIConnectionError, InternalServerError, RateLimitError, NoChoices, json.JSONDecodeError) as e:
            if attempt == attempts - 1 or isinstance(e, APITimeoutError):
                raise
            await asyncio.sleep(10 * 2**attempt)
    raise AssertionError("unreachable")  # the loop returns or raises
