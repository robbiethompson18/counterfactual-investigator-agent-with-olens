"""Gemini 3.8 Flash, OpenRouter first, Google AI Studio direct (`GEMINI_API_KEY`) when OpenRouter says 402/429.

Returns `(text, info)`: the reply text and where it came from (`route`, `key`, `effort`, the thinking text if any).
The direct API rejects thinkingLevel "minimal" on gemini-3.8-flash (400 "Thinking level MINIMAL is not supported",
2026-10-28), so a `minimal` call that falls back runs at "low" and says so in `info`.
"""

import asyncio
import os
from typing import Any

import httpx
from openai import APIStatusError, RateLimitError

from counterfactual_investigator_agent_with_olens import openrouter

MODEL = "google/gemini-3.8-flash"
TIMEOUT = 180  # the direct key hung >120 s on 2026-10-27/28
GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"


async def google(system: str | None, user: str, effort: str, temperature: float, schema: dict | None) -> tuple[str, dict]:
    level = "low" if effort == "minimal" else effort
    body: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": temperature, "thinkingConfig": {"thinkingLevel": level, "includeThoughts": True}},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if schema:
        body["generationConfig"] |= {"responseMimeType": "application/json", "responseJsonSchema": schema}
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for attempt in range(4):
            try:
                r = await client.post(GOOGLE_URL, params={"key": os.environ["GEMINI_API_KEY"]}, json=body)
                if r.status_code in (429, 500, 503) and attempt < 3:
                    await asyncio.sleep(15 * 2**attempt)
                    continue
                r.raise_for_status()
                break
            except httpx.TimeoutException:
                if attempt == 3:
                    raise
    parts = r.json()["candidates"][0]["content"]["parts"]
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    thoughts = "".join(p.get("text", "") for p in parts if p.get("thought"))
    return text, {"route": "google-direct", "key": "GEMINI_API_KEY", "effort": level, "reasoning": thoughts or None}


async def chat(
    user: str, *, system: str | None = None, effort: str, temperature: float, schema: dict | None = None, **params: Any
) -> tuple[str, dict]:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    if schema:
        params |= {"response_format": {"type": "json_schema", "json_schema": schema}}
    try:
        r = await openrouter.chat(MODEL, messages, effort=effort, temperature=temperature, timeout=TIMEOUT, attempts=3, **params)
    except (RateLimitError, APIStatusError) as e:
        if getattr(e, "status_code", None) not in (402, 429):
            raise
        return await google(system, user, effort, temperature, schema["schema"] if schema else None)
    m = r.choices[0].message
    return m.content or "", {
        "route": "openrouter",
        "key": "OPENROUTER_API_KEY",
        "effort": effort,
        "provider": getattr(r, "provider", None),
        "reasoning": getattr(m, "reasoning", None),
    }
