"""Agam's scripts/run_chive.py with every model call routed through OpenRouter.

His harness talks to Anthropic directly (native Messages API for the agent, the OpenAI-compatible
endpoint for the judge). We only have an OpenRouter key, and we want Gemini 3.8 Flash as the
agent. So this loads his script unchanged and swaps three things before running it:
  - the agent backend -> OpenRouter, AGENT_MODEL (and the refusal fallback)
  - the judge's client -> OpenRouter, model ids prefixed `anthropic/` (judge stays Opus 5); same
    for the top20 and concluder stages and the resample executor's grader
Stages, arms, prompts, record layout and reports are all his. Usage is the same as his script:

    OPENROUTER_API_KEY=... uv run python scripts/run_chive_or.py stage=agent eval=smoke_free \\
        ids=splits/smoke25.txt out=data/runs/chive draws=1 "variants=['textfirst','absence']" \\
        "arms=['transcript','olens_rl_B=data/banks/olens_rl:olens']"
"""

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

from detective_joracle.agent.backends import async_anthropic_backend, async_openai_compatible_backend
from detective_joracle.chive import executor, judge, llm
from detective_joracle.chive.llm import anthropic_client
from openai import AsyncOpenAI

# An OpenRouter id (google/gemini-3.8-flash) runs on OpenRouter; an Anthropic id (claude-sonnet-5-5)
# runs on Agam's native Anthropic backend with ANTHROPIC_API_KEY, at his effort setting.
AGENT_MODEL = os.environ.get("CHIVE_AGENT", "google/gemini-3.8-flash")
# Agam runs Opus at effort "max"; OpenRouter's reasoning.effort tops out at "high".
AGENT_EFFORT = "high"
SCRIPT = Path(__file__).resolve().parents[1] / "vendor/detective-joracle/scripts/run_chive.py"


def patch(module: Any, **attrs: Any) -> None:
    """setattr that fails if upstream renamed the attribute, instead of the patch silently not applying."""
    for name, value in attrs.items():
        assert hasattr(module, name), f"{module.__name__}.{name} no longer exists upstream; update scripts/run_chive_or.py"
        setattr(module, name, value)


def openrouter(max_retries: int = 10) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1", max_retries=max_retries)


def agent_backend(max_tokens: int, effort: str, model: str = AGENT_MODEL) -> Any:
    if model.startswith("claude-"):
        inner = async_anthropic_backend(anthropic_client(), model, max_tokens=max_tokens, effort=effort or None)

        async def call(messages: list[dict[str, Any]], tools: list[dict[str, Any]], force_tool: str | None = None) -> Any:
            # Sonnet 5.5 rejects a forced tool_choice with a 400; the loop's nudge message still asks for the call
            return await inner(messages, tools, None)

        return call
    extra: dict[str, Any] = {"usage": {"include": True}}
    if "gemini" in model:
        # same as the harness's own Gemini settings (agent/backends.py::_gemini_extra), effort aside:
        # content filters must not decide an episode
        extra["safety_settings"] = [
            {"category": c, "threshold": "BLOCK_NONE"}
            for c in (
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            )
        ]
    return async_openai_compatible_backend(openrouter(), model, extra_body=extra, max_tokens=max_tokens, effort=AGENT_EFFORT)


_json_call = judge.json_call


async def judge_json_call(c: AsyncOpenAI, *, model: str, **kwargs: Any) -> dict[str, Any] | None:
    return await _json_call(c, model=model if "/" in model else f"anthropic/{model}", **kwargs)


patch(judge, client=openrouter, json_call=judge_json_call)
patch(executor, client=openrouter, json_call=judge_json_call)  # the resample grader (experiments=...)


def schemas(*args: Any, edits: list[str] | None = None, **kwargs: Any) -> list[dict[str, Any]]:
    """His schemas, with `predictions` declaring each edit id as a property. As a bare
    {"type": "object"}, Gemini's function calling drops its keys: Flash submitted `predictions: {}`
    until max_calls in 15/25 smoke yes/no O-lens episodes, which then had no answer."""
    out = _schemas(*args, edits=edits, **kwargs)
    for f in out:
        pred = f["function"]["parameters"]["properties"].get("predictions")
        if pred is not None and edits:
            pred |= {"properties": {e: {"type": "number"} for e in edits}, "required": list(edits)}
    return out


spec = importlib.util.spec_from_file_location("run_chive", SCRIPT)
assert spec and spec.loader
run_chive: Any = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_chive)
_schemas = run_chive.schemas
# His fallback (Sonnet 5) covers Opus content-filter refusals; we just retry Flash once. Side
# effects: every record's `refused_by` reads [AGENT_MODEL] (ignore it), and a judging that gets no
# Opus verdict at all falls back to this model (recorded as `judge_model`).
# The top20 and concluder stages call client/json_call directly rather than through judge.
patch(
    run_chive,
    AGENT_MODEL=AGENT_MODEL,
    FALLBACK_MODEL=AGENT_MODEL,
    BACKEND="openrouter",
    agent_backend=agent_backend,
    schemas=schemas,
    client=openrouter,
    json_call=judge_json_call,
)
# Any module that imported llm.client / llm.json_call by value and isn't patched above would still call Anthropic.
unpatched = [
    f"{name}.{attr}"
    for name, module in list(sys.modules.items())
    if name.startswith("detective_joracle") and module is not llm
    for attr in ("client", "json_call")
    if getattr(module, attr, None) in (llm.client, llm.json_call)
]
assert not unpatched, f"unpatched Anthropic calls in {unpatched}; patch them in scripts/run_chive_or.py"

if __name__ == "__main__":
    run_chive.main()
