"""Donor assistant turns: Gemma-3-27B and Gemini-2.5-Flash (OpenRouter) and Qwen3.6-27B itself (vLLM) answering
the same TASKS + PRESSURE user turns. Scrubbed of self-naming. -> data/runs/identity/2026-10-21_env_google/donors.json

    uv run --with openai --with anthropic --with transformers --with jinja2 python scripts/identity/env_google/gen_donors.py [or|qwen]
"""

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "scripts/identity")
sys.path.insert(0, "scripts/identity/env_google")
from prompts import PRESSURE, TASKS
from qwen import openrouter, or_spent, sample_many

OUT = Path("data/runs/identity/2026-10-21_env_google/donors.json")
# AI labs/models (any case) plus third-party products seen in the donor replies (case-sensitive), since
# Robbie's rule is no vendor/product names anywhere in a cue.
NAMES = re.compile(
    r"(?i:\b(?:Gemma|Gemini|Google|DeepMind|Bard|Qwen|Alibaba|Tongyi|OpenAI|ChatGPT|GPT|Claude|Anthropic|Meta AI|Llama)\b)"
    r"|\b(?:Microsoft|Apple|Amazon|AWS|Azure|Slack|Zoom|Asana|Trello|Notion|Spotify|Airbnb|Uber|Netflix|Facebook|Instagram|Android|iPhone|Kelley|Edmunds|Carfax|Headspace)\b"
)
DONORS = {"gemma": "google/gemma-3-27b-it", "gemini": "google/gemini-2.5-flash"}
MAX_TOKENS = 900


def scrub(text):
    """Drop any sentence that names a model or lab (rare; the tasks never ask about identity)."""
    return re.sub(r"[^.!?\n]*(?:" + NAMES.pattern + r")[^.!?\n]*[.!?]?[ \t]*", "", text).strip()


def main(which):
    donors = json.loads(OUT.read_text()) if OUT.exists() else {}
    prompts = TASKS + PRESSURE
    if which == "rescrub":
        for v in donors.values():
            for x in v:
                x["assistant"] = scrub(x["raw"])
    elif which == "or":
        print("spent before", or_spent())
        for name, model in DONORS.items():
            with ThreadPoolExecutor(8) as ex:
                outs = list(
                    ex.map(lambda p: openrouter(model, [{"role": "user", "content": p}], who="env_google", max_tokens=MAX_TOKENS), prompts)
                )
            donors[name] = [{"user": p, "raw": o, "assistant": scrub(o)} for p, o in zip(prompts, outs)]
        print("spent after", or_spent())
    else:
        outs = sample_many([[{"role": "user", "content": p}] for p in prompts], n=1, thinking=True, max_tokens=4096)
        donors["qwen"] = [{"user": p, "raw": o[0]["content"], "assistant": scrub(o[0]["content"] or "")} for p, o in zip(prompts, outs)]
    OUT.write_text(json.dumps(donors, indent=1))
    for k, v in donors.items():
        L = sorted(len(x["assistant"]) for x in v)
        print(k, "n", len(v), "median chars", L[len(L) // 2], "scrubbed", sum(x["raw"] != x["assistant"] for x in v))


main(sys.argv[1])
