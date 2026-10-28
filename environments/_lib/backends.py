"""Model backends, chosen per run by flags (never by env vars):

    vllm            any OpenAI-compatible URL (the shared Qwen server, or a pod started for one sweep)
    openrouter      provider pinned (--provider) with no fallbacks, reasoning effort always sent
    openrouter-raw  GLM 5.x thinking off: the chat template rendered here, an empty think block prefilled, raw completions

`Target` is everything that decides which model answered; its `id` names a deployment, so runs from different deployments
are never pooled. Every call returns the provider that actually served it.
"""

import asyncio
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx

OPENROUTER = "https://openrouter.ai/api/v1"


# a vLLM server reached through a local port (an ssh tunnel to the pod) says nothing about the deployment, so its id is
# `@vllm:any`, as the converted runs have it; runs made before that rule stored the loopback host (canonical_id maps them)
LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def canonical_id(target_id: str | None) -> str | None:
    """A stored target id with a loopback host (`Qwen/Qwen3.6-27B@vllm:127.0.0.1`) as `...@vllm:any`, so its runs pool."""
    if not target_id:
        return target_id
    model, _, rest = target_id.rpartition("@")
    backend, sep, where = rest.partition(":")
    return f"{model}@{backend}:any" if sep and where in LOOPBACK else target_id


@dataclass
class Target:
    model: str  # e.g. z-ai/glm-5.3, Qwen/Qwen3.6-27B (the name the server is asked for: --served-name)
    backend: str  # vllm | openrouter | openrouter-raw
    url: str | None = None  # vllm base URL
    served_name: str | None = None  # model name to send, if the server's differs from --model (the shared server says "qwen")
    provider: str | None = None  # OpenRouter provider slug, pinned
    effort: str | None = "medium"  # reasoning effort when thinking is on; None only if the env REQUIRES it
    thinking: bool = False  # Qwen default is thinking off (CLAUDE.md); envs pin their own in DEFAULTS
    max_tokens: int | None = None  # default no cap (repo rule for Qwen); a cap is recorded in meta
    sampling: dict = field(default_factory=dict)  # temperature / top_p / top_k / presence_penalty, only if given
    # Request body layout, so a ported env sends exactly what its old harness sent:
    #   "openrouter"  evalcue / src openrouter.extra_body: sampling, usage accounting, reasoning {enabled, effort when
    #                 thinking}, provider {"order": [p], "allow_fallbacks": false}; for vllm, Qwen's chat_template_kwargs
    #   "realism"     donehack/env.py chat: model, messages, tools, reasoning {"effort"} and nothing else
    body: str = "openrouter"

    @property
    def id(self) -> str:
        host = urlparse(self.url).hostname if self.url else None
        where = self.provider or (host if host not in LOOPBACK else None) or "any"
        return f"{self.model}@{self.backend}:{where}"

    def record(self) -> dict:
        d = asdict(self)
        d["url"] = urlparse(self.url).hostname if self.url else None  # host only: URLs can carry tokens
        d["id"] = self.id
        return d


def openrouter_key() -> str:
    for src in (os.environ.get("OPENROUTER_API_KEY"), Path("/secrets/openrouter")):
        if isinstance(src, str) and src:
            return src
        if isinstance(src, Path) and src.exists():
            return src.read_text().strip()
    raise SystemExit("no OpenRouter key: set OPENROUTER_API_KEY")


def vllm_key() -> str:
    return os.environ.get("VLLM_API_KEY") or os.environ.get("QWEN_VLLM_API_KEY") or "EMPTY"


# zai-org/GLM-5.3 chat template, single turn, with an empty think block (copied from evalcue/harness.py's raw-prefill path)
GLM_TURN = {"system": "<|system|>", "user": "<|user|>"}
GLM_PREFIX, GLM_EMPTY_THINK = "[gMASK]<sop><|system|>Reasoning Effort: Max", "<|assistant|><think></think>"
RAW_PROVIDERS = ["fireworks", "together", "parasail/fp8", "friendli"]


class Transient(Exception):
    pass


async def chat(client: httpx.AsyncClient, t: Target, messages: list[dict], tools: list[dict] | None = None) -> dict:
    """One completion. -> {"message": {...}, "finish": str, "provider": str | None, "usage": {...}}"""
    if t.backend == "openrouter-raw":
        assert not tools and all(m["role"] in GLM_TURN for m in messages), "openrouter-raw: GLM, single turn, no tools"
        prompt = GLM_PREFIX + "".join(GLM_TURN[m["role"]] + m["content"] for m in messages) + GLM_EMPTY_THINK
        body = {
            "model": t.served_name or t.model,
            "prompt": prompt,
            "max_tokens": t.max_tokens or 16384,  # raw completions default to a few tokens on some providers
            **t.sampling,
            "provider": {"only": [t.provider] if t.provider else RAW_PROVIDERS, "allow_fallbacks": False},
        }  # as evalcue's chat_raw_prefill sent it
        j = await _post(client, f"{OPENROUTER}/completions", body, openrouter_key())
        c = j["choices"][0]
        msg = {
            "role": "assistant",
            "content": c.get("text") or "",
            "reasoning_content": c.get("reasoning") or "",
            "provider": j.get("provider"),
        }
        return {"message": msg, "finish": c.get("finish_reason"), "provider": j.get("provider"), "usage": j.get("usage") or {}}
    body, url, key = request(t, messages, tools)
    j = await _post(client, url, body, key)
    c = j["choices"][0]
    m = c["message"]
    if t.body == "realism":  # the old loop sent back the assistant message as returned (minus None fields and refusal)
        raw = {k: v for k, v in m.items() if v is not None and k != "refusal"}
        provider = j.get("provider")
        return {"message": raw, "finish": c.get("finish_reason"), "provider": provider, "usage": j.get("usage") or {}}
    msg = {"role": "assistant", "content": m.get("content") or ""}
    reasoning = m.get("reasoning_content") or m.get("reasoning")
    if reasoning:
        msg["reasoning_content"] = reasoning
    if m.get("tool_calls"):
        msg["tool_calls"] = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["function"]["name"], "arguments": tc["function"].get("arguments") or ""},
            }
            for tc in m["tool_calls"]
        ]
    provider = j.get("provider") or (urlparse(t.url).hostname if t.url else None)
    return {"message": msg, "finish": c.get("finish_reason"), "provider": provider, "usage": j.get("usage") or {}}


def request(t: Target, messages: list[dict], tools: list[dict] | None) -> tuple[dict, str, str]:
    """-> (JSON body, URL, API key) for one chat call. Pure, so a stored episode's first request can be rebuilt and
    compared with what its old harness sent (environments/_lib/replay.py)."""
    if t.backend == "openrouter" and t.body == "realism":
        body = {"model": t.served_name or t.model, "messages": messages, "tools": tools, **t.sampling}
        if t.max_tokens:
            body["max_tokens"] = t.max_tokens
        if t.thinking:
            body["reasoning"] = {"effort": t.effort} if t.effort else {}
        else:
            body["reasoning"] = {"enabled": False}
        if t.provider:
            body["provider"] = {"order": [t.provider], "allow_fallbacks": False}
        return {k: v for k, v in body.items() if v is not None}, f"{OPENROUTER}/chat/completions", openrouter_key()
    body: dict = {"model": t.served_name or t.model, "messages": messages}
    if t.max_tokens:
        body["max_tokens"] = t.max_tokens
    if tools:
        body["tools"] = tools
    body |= t.sampling
    if t.backend == "openrouter":
        body["usage"] = {"include": True}
        body["reasoning"] = {"enabled": t.thinking, **({"effort": t.effort} if t.thinking and t.effort else {})}
        if t.provider:
            body["provider"] = {"order": [t.provider], "allow_fallbacks": False}
        return body, f"{OPENROUTER}/chat/completions", openrouter_key()
    if t.backend == "vllm":
        if not t.url:
            raise SystemExit("--backend vllm needs --url")
        body["chat_template_kwargs"] = {"enable_thinking": t.thinking, "preserve_thinking": True}
        return body, t.url.rstrip("/") + "/chat/completions", vllm_key()
    raise SystemExit(f"unknown backend {t.backend}")


async def _post(client, url, body, key, tries=6) -> dict:
    for attempt in range(tries):
        try:
            r = await client.post(url, json=body, headers={"Authorization": f"Bearer {key}"})
            if r.status_code in (408, 429) or r.status_code >= 500:
                raise Transient(f"HTTP {r.status_code}: {r.text[:300]}")
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
            j = r.json()
            if not j.get("choices"):  # OpenRouter reports some provider errors as a 200 with no choices
                raise Transient(f"no choices: {str(j)[:300]}")
            return j
        except (Transient, httpx.TransportError) as e:
            if attempt == tries - 1:
                raise RuntimeError(f"{url}: gave up after {tries} tries: {e}") from e
            await asyncio.sleep(10 * 2**attempt)
    raise AssertionError
