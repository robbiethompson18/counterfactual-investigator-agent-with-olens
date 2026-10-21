"""Shared client for the identity-belief experiments: Qwen3.6-27B samples (vLLM), O-lens reads, the
Sonnet 5.5 judge, and a budget-capped OpenRouter call. Needs scripts/identity/tunnel.sh running.

    uv run --with openai --with anthropic --with transformers --with jinja2 python your_script.py

    import sys; sys.path.insert(0, "scripts/identity"); from qwen import *
    outs = sample_many([msgs1, msgs2], n=8)              # [[{reasoning, content, finish_reason}] * 8] * 2
    ids, toks = render(msgs1, outs[0][0])                # chat-templated token ids of prompt + that reply
    reads = read(ids, [(p, L) for p in header_positions(ids) for L in (32, 40, 48)])
"""

import asyncio
import json
import os
import re
import time
from functools import cache
from pathlib import Path

import requests
from openai import AsyncOpenAI

VLLMS = ["http://127.0.0.1:28000/v1", "http://127.0.0.1:28001/v1"]  # replicas; sample_many spreads over the live ones
LENS = "http://127.0.0.1:28802"
OR_LEDGER = Path("data/runs/identity/openrouter_ledger.jsonl")
OR_BUDGET_USD = 10.0
LAYERS = tuple(range(20, 61, 4))  # the O-lens reads layers 20..60 step 4


@cache
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")


# ---------- sampling ----------


async def _sample(client, sem, messages, n, thinking, temperature, max_tokens, **kw):
    # Qwen's recommended settings: thinking T=0.6/top_p=0.95, non-thinking T=0.7/top_p=0.8; top_k=20
    temperature = temperature if temperature is not None else (0.6 if thinking else 0.7)
    extra = {"top_k": 20, "chat_template_kwargs": {"enable_thinking": thinking}}
    async with sem:
        r = await client.chat.completions.create(
            model="qwen",
            messages=messages,
            n=n,
            temperature=temperature,
            top_p=0.95 if thinking else 0.8,
            max_tokens=max_tokens,
            extra_body=extra,
            **kw,
        )
    return [
        {
            "reasoning": getattr(c.message, "reasoning_content", None) or getattr(c.message, "reasoning", None),
            "content": c.message.content,
            "finish_reason": c.finish_reason,
        }
        for c in r.choices
    ]


def _up(url):
    try:
        return requests.get(url + "/models", timeout=5).ok
    except requests.RequestException:
        return False


def sample_many(conversations, n=1, thinking=True, temperature=None, max_tokens=4096, concurrency=64, **kw):
    """conversations: list of OpenAI-style message lists -> per conversation, n samples."""

    live = [u for u in VLLMS if _up(u)]
    assert live, f"no vLLM replica answering at {VLLMS}"

    async def go():
        clients = [AsyncOpenAI(base_url=u, api_key="none", max_retries=3, timeout=900) for u in live]
        sem = asyncio.Semaphore(concurrency)
        return await asyncio.gather(
            *[_sample(clients[i % len(clients)], sem, m, n, thinking, temperature, max_tokens, **kw) for i, m in enumerate(conversations)]
        )

    return asyncio.run(go())


# ---------- rendering + positions ----------


def render(messages, reply=None, thinking=True):
    """Token ids (and token strings) of the chat-templated conversation, plus `reply` (a sample dict, or
    a plain string = content with no reasoning) as the assistant turn. reply=None: prompt only, ending
    at the generation prompt (the assistant header, and `<think>\\n` when thinking)."""
    t = tok()
    text = t.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    if reply is not None:
        if isinstance(reply, str):
            reply = {"reasoning": None, "content": reply}
        if thinking and not text.rstrip().endswith("<think>"):
            text += "<think>\n"
        if thinking:
            text += (reply.get("reasoning") or "").strip() + "\n</think>\n\n"
        text += (reply.get("content") or "") + "<|im_end|>"
    ids = t(text, add_special_tokens=False).input_ids
    return ids, [t.decode([i]) for i in ids]


def header_positions(ids):
    """The end-of-user-turn zone: the last <|im_end|> before the assistant turn through the assistant
    header (and `<think>`). The lens reads best here (docs/model-organisms.md)."""
    t = tok()
    end, start = t.convert_tokens_to_ids("<|im_end|>"), t.convert_tokens_to_ids("<|im_start|>")
    a = max(i for i, x in enumerate(ids) if x == start)  # last <|im_start|> = the assistant header
    u = max(i for i in range(a) if ids[i] == end)
    stop = a + 4
    return list(range(u, min(stop, len(ids))))


def find_positions(toks, needle, last=False):
    """Token indices whose decoded text completes `needle` (first char match on the joined text)."""
    joined, starts = "", []
    for s in toks:
        starts.append(len(joined))
        joined += s
    hits = [m.end() - 1 for m in re.finditer(re.escape(needle), joined)]
    pos = [max(i for i, s in enumerate(starts) if s <= h) for h in hits]
    return pos[-1:] if last and pos else pos


# ---------- O-lens reads ----------


def read(ids, items, temperature=0.0, n=1, seed=0, max_bullets=4):
    """items: [(pos, layer)] -> per item, n readouts {text, bullets, bfve (per-bullet whitened FVE), fve}.
    One activation capture for the whole sequence. The server is shared and serial: keep requests to
    a few hundred items, and long contexts cost a full forward pass each call."""
    body = {
        "ids": list(ids),
        "items": [{"pos": int(p), "layer": int(L)} for p, L in items],
        "temperature": temperature,
        "n": n,
        "seed": seed,
        "max_bullets": max_bullets,
    }
    for attempt in range(30):
        r = requests.post(LENS + "/read", json=body, timeout=3600)
        if r.status_code == 503:  # still loading
            time.sleep(30)
            continue
        r.raise_for_status()
        return r.json()["results"]
    raise RuntimeError("lens server never became ready")


def lens_status():
    return requests.get(LENS + "/status", timeout=10).json()


# ---------- grading ----------
# No LLM-API graders for these exploratory runs (CLAUDE.local.md): label with regex, read the
# transcripts yourself, or hand batches to Sonnet/Haiku subagents inside Claude Code.


def judge(*_a, **_k):
    raise RuntimeError("no API graders: use regex, read transcripts yourself, or a Sonnet/Haiku subagent (CLAUDE.local.md)")


def judge_json(text):
    """Last {...} block of a text, parsed (e.g. a subagent's label output)."""
    m = re.findall(r"\{.*\}", text, re.DOTALL)
    return json.loads(m[-1]) if m else None


# ---------- OpenRouter, capped at OR_BUDGET_USD across all agents (shared ledger) ----------


def or_spent():
    return sum(json.loads(line)["cost"] for line in OR_LEDGER.open()) if OR_LEDGER.exists() else 0.0


def openrouter(model, messages, who, **kw):
    """One chat completion from OpenRouter (e.g. model="google/gemma-3-27b-it"); logs its cost to the
    shared ledger and refuses once the ledger passes the budget. Returns the message content."""
    spent = or_spent()
    if spent >= OR_BUDGET_USD:
        raise RuntimeError(f"OpenRouter budget spent: ${spent:.2f} of ${OR_BUDGET_USD}")
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        json={"model": model, "messages": messages, "usage": {"include": True}, **kw},
        timeout=600,
    )
    r.raise_for_status()
    j = r.json()
    OR_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with OR_LEDGER.open("a") as f:
        f.write(json.dumps({"t": time.time(), "who": who, "model": model, "cost": float(j.get("usage", {}).get("cost") or 0)}) + "\n")
    return j["choices"][0]["message"]["content"]
