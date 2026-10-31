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
OpenAI-compatible API with the key in $`key_env` (default GEMINI_API_KEY; OLD_GEMINI_API_KEY, THIRD_GEMINI_KEY, ...), at
`reasoning_effort` = effort, so a run can use a key whose daily quota isn't the one OpenRouter's BYOK route is spending. The
only direct-to-Google client: don't hand-roll generateContent. No `usage.cost`.
`vertex/<model>` does the same on Vertex AI's OpenAI-compatible endpoint in $VERTEX_PROJECT, authenticated as the active
gcloud account (token re-read every 2 min); for when AI Studio 503s "high demand" but Vertex has capacity.
`anthropic-direct/<model>` (e.g. `anthropic-direct/claude-haiku-5-5`, Anthropic's id, not OpenRouter's) never touches
OpenRouter, whose BYOK route falls back to OpenRouter's own paid capacity when our key is rate-limited: it calls Anthropic's
Messages API with $ANTHROPIC_API_KEY at `output_config.effort` = effort (`off` sends `thinking: disabled`), translating
OpenAI-format messages and tools in and the reply back into a ChatCompletion (`completion_tokens_details.reasoning_tokens`
= thinking tokens; `max_tokens` defaults to 16000). The reply's message keeps Anthropic's raw content blocks in
`anthropic_content` (thinking signatures included), so appending `message.model_dump()` replays them next turn. No `usage.cost`.
`anthropic-batch/<model>` is the same request through the Message Batches API: half price and outside the per-minute token
limit (2M input tokens/min on Haiku 5.5 at our tier), but a reply takes minutes to hours. Calls are queued, sent together
once BATCH_WAIT s pass (at most BATCH_MAX per batch) and polled until their batch ends, so it pays only with many concurrent
calls (e.g. cfbench.eval at --concurrency 5000). An errored or expired result is retried like a 5xx.
"""

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import anthropic
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, InternalServerError, RateLimitError
from openai.types.chat import ChatCompletion

BASE_URL = "https://openrouter.ai/api/v1"
GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DIRECT = "gemini-direct/"
VERTEX = "vertex/"
CLAUDE = "anthropic-direct/"
BATCH = "anthropic-batch/"
BATCH_WAIT, BATCH_MAX, BATCH_POLL = 30, 1000, 30  # 1,000 inline prompts of ~25k tokens is ~100 MB, under the 256 MB batch cap
EFFORT = "medium"
_CLIENT: AsyncOpenAI | None = None
_GOOGLE: dict[str, AsyncOpenAI] = {}  # key_env -> client
_VERTEX: tuple[float, AsyncOpenAI] | None = None  # (built at, client): gcloud returns its cached hour-long token until it nears expiry
_ANTHROPIC: anthropic.AsyncAnthropic | None = None
_QUEUE: list[tuple[dict, asyncio.Future]] = []  # anthropic-batch/ requests waiting for the next batch
_BATCHER: asyncio.Task | None = None
_BATCHES: set[asyncio.Task] = set()  # running batches, held so they aren't garbage-collected
# 429, 529 overloaded, other 5xx, connection errors (timeouts are a subclass, re-raised below)
ANTHROPIC_TRANSIENT = (anthropic.RateLimitError, anthropic.OverloadedError, anthropic.InternalServerError, anthropic.APIConnectionError)


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


def claude_request(model: str, messages: list[Any], effort: str | None, **params: Any) -> dict:
    """A Messages API request from `chat`'s OpenAI shapes (module docstring). Consecutive user turns (tool results, then a
    nudge) are left for the API to merge."""
    turns: list[Any] = []  # plain dicts, as for the OpenAI SDK
    for m in messages:
        if m["role"] == "assistant":
            calls = [
                {"type": "tool_use", "id": c["id"], "name": c["function"]["name"], "input": json.loads(c["function"]["arguments"])}
                for c in m.get("tool_calls") or []
            ]
            text = [{"type": "text", "text": m["content"]}] if m.get("content") else []
            turns.append({"role": "assistant", "content": m.get("anthropic_content") or text + calls})
        elif m["role"] == "tool":
            turns.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}]})
        elif m["role"] == "user":
            turns.append({"role": "user", "content": m["content"]})
    if system := "\n\n".join(m["content"] for m in messages if m["role"] == "system"):
        params["system"] = system
    if tools := params.pop("tools", None):
        params["tools"] = [
            {"name": f["name"], "description": f.get("description", ""), "input_schema": f.get("parameters") or {"type": "object"}}
            for f in (t["function"] for t in tools)
        ]
    if effort == "off":
        params["thinking"] = {"type": "disabled"}
    elif effort:
        params["output_config"] = {"effort": effort}
    return {"model": model, "messages": turns, "max_tokens": params.pop("max_tokens", 16000), **params}


def claude_reply(r: anthropic.types.Message) -> ChatCompletion:
    """An Anthropic reply as a ChatCompletion (module docstring)."""
    uses = [b for b in r.content if b.type == "tool_use"]
    calls = [{"id": b.id, "type": "function", "function": {"name": b.name, "arguments": json.dumps(b.input)}} for b in uses]
    message = {
        "role": "assistant",
        "content": "".join(b.text for b in r.content if b.type == "text") or None,
        "tool_calls": calls or None,
        "anthropic_content": [b.model_dump(exclude_none=True) for b in r.content],
    }
    u = r.usage
    thinking = u.output_tokens_details.thinking_tokens if u.output_tokens_details else 0
    prompt = u.input_tokens + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)
    finish = {"max_tokens": "length", "tool_use": "tool_calls", "refusal": "content_filter"}.get(r.stop_reason or "", "stop")
    return ChatCompletion.model_validate(
        {
            "id": r.id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": r.model,
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": u.output_tokens,
                "total_tokens": prompt + u.output_tokens,
                "completion_tokens_details": {"reasoning_tokens": thinking},
            },
        }
    )


def anthropic_client(timeout: float = 600) -> anthropic.AsyncAnthropic:
    """The shared Anthropic client ($ANTHROPIC_API_KEY), built on first use."""
    global _ANTHROPIC
    if _ANTHROPIC is None:
        _ANTHROPIC = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=timeout, max_retries=2)
    return _ANTHROPIC


async def _batched(api: anthropic.AsyncAnthropic, request: dict) -> anthropic.types.Message:
    """`request` answered through the Message Batches API (module docstring)."""
    global _BATCHER
    future = asyncio.get_running_loop().create_future()
    _QUEUE.append((request, future))
    if _BATCHER is None:
        _BATCHER = asyncio.create_task(_send_batches(api))
    return await future


async def _send_batches(api: anthropic.AsyncAnthropic) -> None:
    global _BATCHER
    while _QUEUE:
        await asyncio.sleep(BATCH_WAIT)
        while _QUEUE:
            chunk = _QUEUE[:BATCH_MAX]
            del _QUEUE[:BATCH_MAX]
            task = asyncio.create_task(_run_batch(api, chunk))
            _BATCHES.add(task)
            task.add_done_callback(_BATCHES.discard)
    _BATCHER = None


async def _run_batch(api: anthropic.AsyncAnthropic, chunk: list[tuple[dict, asyncio.Future]]) -> None:
    try:
        requests: list[Any] = [{"custom_id": str(i), "params": request} for i, (request, _) in enumerate(chunk)]
        batch = await api.messages.batches.create(requests=requests)
        print(f"anthropic batch {batch.id}: {len(chunk)} requests", flush=True)
        while batch.processing_status != "ended":
            await asyncio.sleep(BATCH_POLL)
            batch = await api.messages.batches.retrieve(batch.id)
        async for entry in await api.messages.batches.results(batch.id):
            future, result = chunk[int(entry.custom_id)][1], entry.result
            if result.type == "succeeded":
                future.set_result(result.message)
            else:
                future.set_exception(NoChoices(f"batch {batch.id} {result.type}: {getattr(result, 'error', '')}"[:500]))
        for _, future in chunk:
            if not future.done():
                future.set_exception(NoChoices(f"batch {batch.id}: no result"))
    except Exception as e:  # noqa: BLE001 -- anything, or the callers wait forever; a 429 or 5xx creating the batch is retried
        for _, future in chunk:
            if not future.done():
                future.set_exception(e)


async def chat(
    model: str,
    messages: list[Any],  # plain dicts, as the SDK accepts them
    *,
    effort: str | None = EFFORT,
    provider: dict | None = None,
    key_env: str = "GEMINI_API_KEY",
    timeout: float = 600,
    attempts: int = 6,
    **params: Any,
) -> ChatCompletion:
    """One chat completion with thinking on at `effort`. Connection errors, 5xx, 429, empty-choices and non-JSON responses are retried
    `attempts` times with 10 s, 20 s, 40 s, ... between tries (~5 min in all for 6, which outlasts OpenRouter's per-minute
    429s); a timeout is not retried, it already waited `timeout` seconds. Other request fields go in `params`; an
    `extra_body` there is merged over the OpenRouter fields. `key_env` names the env var holding the key for `gemini-direct/`."""
    global _CLIENT, _VERTEX
    if model.startswith(VERTEX):
        if _VERTEX is None or time.time() - _VERTEX[0] > 120:  # so re-read often, or the cached token runs out under us
            cmd = ["gcloud", "auth", "print-access-token"]
            token = str(await asyncio.to_thread(subprocess.check_output, cmd, text=True)).strip()
            url = f"https://aiplatform.googleapis.com/v1/projects/{os.environ['VERTEX_PROJECT']}/locations/global/endpoints/openapi"
            _VERTEX = (time.time(), AsyncOpenAI(base_url=url, api_key=token, timeout=timeout, max_retries=2))
        api, model = _VERTEX[1], "google/" + model.removeprefix(VERTEX)
        extra = params.pop("extra_body", None) or {}
        params["reasoning_effort"] = "none" if effort == "off" else effort
    elif model.startswith(DIRECT):
        if key_env not in _GOOGLE:
            _GOOGLE[key_env] = AsyncOpenAI(base_url=GOOGLE_URL, api_key=os.environ[key_env], timeout=timeout, max_retries=2)
        api, model = _GOOGLE[key_env], model.removeprefix(DIRECT)
        extra = params.pop("extra_body", None) or {}
        params["reasoning_effort"] = "none" if effort == "off" else effort
    elif model.startswith((CLAUDE, BATCH)):
        api, extra = anthropic_client(timeout), {}
    else:
        if _CLIENT is None:  # built on first use, not at import: modules get imported on machines without the key
            _CLIENT = client()
        api = _CLIENT
        extra = extra_body(thinking=effort != "off", effort=effort, provider=provider) | (params.pop("extra_body", None) or {})
    for attempt in range(attempts):
        try:
            if isinstance(api, anthropic.AsyncAnthropic):
                request = claude_request(model.split("/", 1)[1], messages, effort, **params)
                reply = await _batched(api, request) if model.startswith(BATCH) else await api.messages.create(**request, timeout=timeout)
                r = claude_reply(reply)
            else:
                r = await api.chat.completions.create(model=model, messages=messages, timeout=timeout, extra_body=extra, **params)
            if not getattr(r, "choices", None):
                raise NoChoices(str(r.model_extra)[:500])
            return r
        except (APIConnectionError, InternalServerError, RateLimitError, NoChoices, json.JSONDecodeError, *ANTHROPIC_TRANSIENT) as e:
            if attempt == attempts - 1 or isinstance(e, (APITimeoutError, anthropic.APITimeoutError)):
                raise
            await asyncio.sleep(10 * 2**attempt)
    raise AssertionError("unreachable")  # the loop returns or raises
