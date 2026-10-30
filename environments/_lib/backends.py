"""Model backends, chosen per run by flags (never by env vars):

    vllm            any OpenAI-compatible URL (the shared Qwen server, or a pod started for one sweep)
    openrouter      provider pinned (--provider) with no fallbacks, reasoning effort always sent
    openrouter-raw  GLM 5.x thinking off: the chat template rendered here, an empty think block prefilled, raw completions
                    (single-turn prompts, or multi-turn tool episodes via GLM 5.3's own template: glm_prompt / glm_parse)

`Target` is everything that decides which model answered; its `id` names a deployment, so runs from different deployments
are never pooled. Every call returns the provider that actually served it.
"""

import asyncio
import json
import os
import re
import zlib
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


GLM_STOP = ["<|observation|>", "<|user|>", "<|endoftext|>"]


def glm_prompt(messages: list[dict], tools: list[dict]) -> str:
    """GLM 5.3's own chat template (glm53_chat_template.jinja, from zai-org/GLM-5.3) over a multi-turn tool conversation,
    ending in an empty think block: thinking off for agent episodes, which OpenRouter's chat endpoint refuses
    ("Reasoning is mandatory"). Prior turns render with empty think blocks too, as the template does."""
    import jinja2

    env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"], keep_trailing_newline=True)
    env.filters["tojson"] = lambda v, ensure_ascii=True, indent=None: json.dumps(v, ensure_ascii=ensure_ascii, indent=indent)
    tpl = env.from_string((Path(__file__).parent / "glm53_chat_template.jinja").read_text())
    msgs = []
    for m in messages:
        m = {k: v for k, v in m.items() if k not in ("reasoning_content", "reasoning", "provider")}
        if m.get("tool_calls"):
            m["tool_calls"] = [
                {**tc, "function": {**tc["function"], "arguments": _args_dict(tc["function"].get("arguments"))}} for tc in m["tool_calls"]
            ]
        msgs.append(m)
    return tpl.render(messages=msgs, tools=tools, add_generation_prompt=False) + "<|assistant|><think></think>"


def _args_dict(a):
    if isinstance(a, dict):
        return a
    try:
        d = json.loads(a or "{}")
        return d if isinstance(d, dict) else {"_raw": a}
    except json.JSONDecodeError:
        return {"_raw": a}


def glm_parse(text: str, tools: list[dict]) -> tuple[str, list[dict]]:
    """GLM's raw output -> (visible text, OpenAI-style tool calls). Calls are `<tool_call>NAME<arg_key>K</arg_key>
    <arg_value>V</arg_value>...</tool_call>`; a value whose schema type isn't string is JSON-decoded."""
    schema = {t["function"]["name"]: t["function"].get("parameters", {}).get("properties", {}) for t in tools}
    content, _, rest = text.partition("<tool_call>")
    calls = []
    for k, body in enumerate(re.split(r"<tool_call>", rest) if rest or "<tool_call>" in text else []):
        body = body.split("</tool_call>")[0]
        name = body.split("<arg_key>")[0].strip()
        args = {}
        for key, val in re.findall(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", body, re.DOTALL):
            key = key.strip()
            if schema.get(name, {}).get(key, {}).get("type", "string") != "string":
                try:
                    val = json.loads(val)
                except json.JSONDecodeError:
                    pass
            args[key] = val
        calls.append(
            {
                "id": f"call_{k}_{zlib.crc32(body.encode()):08x}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
        )
    return content.strip(), calls


async def _glm_raw_turn(client: httpx.AsyncClient, t: Target, messages: list[dict], tools: list[dict]) -> dict:
    body = {
        "model": t.served_name or t.model,
        "prompt": glm_prompt(messages, tools),
        "max_tokens": t.max_tokens or 16384,
        "stop": GLM_STOP,
        **t.sampling,
        "provider": {"only": [t.provider] if t.provider else RAW_PROVIDERS, "allow_fallbacks": False},
    }
    j = await _post(client, f"{OPENROUTER}/completions", body, openrouter_key())
    c = j["choices"][0]
    text = c.get("text") or ""
    content, calls = glm_parse(text, tools)
    msg = {"role": "assistant", "content": content, "provider": j.get("provider")}
    if c.get("reasoning"):
        msg["reasoning_content"] = c["reasoning"]
    if calls:
        msg["tool_calls"] = calls
    finish = "tool_calls" if calls else c.get("finish_reason")
    return {"message": msg, "finish": finish, "provider": j.get("provider"), "usage": j.get("usage") or {}}


async def chat(client: httpx.AsyncClient, t: Target, messages: list[dict], tools: list[dict] | None = None) -> dict:
    """One completion. -> {"message": {...}, "finish": str, "provider": str | None, "usage": {...}}"""
    if t.backend == "openrouter-raw" and (tools or any(m["role"] not in GLM_TURN for m in messages)):
        return await _glm_raw_turn(client, t, messages, tools or [])
    if t.backend == "openrouter-raw":
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


async def chat_raw(client: httpx.AsyncClient, t: Target, messages: list[dict]) -> dict:
    """One assistant turn as raw text, for episodes whose tools are written into the prompt: vLLM's chat endpoint runs its
    tool parser even when no tools are sent (and drops the call text), so render with /tokenize (the same chat template, no
    tools) and generate with /v1/completions. The caller parses the calls (loop.parse_calls)."""
    if t.backend != "vllm":
        raise SystemExit("tools_in_prompt episodes need --backend vllm")
    base = (t.url or "").rstrip("/").removesuffix("/v1")
    kw = {"enable_thinking": t.thinking, "preserve_thinking": True}
    tb = {"model": t.served_name or t.model, "messages": messages, "add_generation_prompt": True, "chat_template_kwargs": kw}
    rt = await client.post(f"{base}/tokenize", json=tb, headers={"Authorization": f"Bearer {vllm_key()}"})
    rt.raise_for_status()
    tok = rt.json()  # not _post: it retries replies without "choices"
    body = {"model": t.served_name or t.model, "prompt": tok["tokens"], **t.sampling}
    body["max_tokens"] = t.max_tokens or tok["max_model_len"] - tok["count"]
    j = await _post(client, f"{base}/v1/completions", body, vllm_key())
    c = j["choices"][0]
    return {
        "message": {"role": "assistant", "content": c.get("text") or ""},
        "finish": c.get("finish_reason"),
        "provider": urlparse(t.url).hostname,
        "usage": j.get("usage") or {},
    }


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
