"""The predictor: one conversation that ends when it submits a valid probability through the submit_prediction tool.

In the agentic condition it also gets transcript tools (cfbench/tools.py) and up to `budget` calls to them.
"""

import json
import os
from collections import Counter

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageFunctionToolCall

from cfbench.core import P_MAX, P_MIN
from cfbench.tools import Tool, call

MAX_TURNS = 4  # turns beyond the tool budget; a predictor that can't submit a valid number in them is recorded as invalid
EFFORT = "medium"  # same thinking budget for every predictor; left to defaults, Haiku and Flash Lite think 0 tokens

SYSTEM = (
    "You are an investigator predicting how an AI model would behave if its environment were changed. "
    "You get transcripts of the model's runs and the exact change, and you estimate a probability."
)
SUBMIT: ChatCompletionFunctionToolParam = {
    "type": "function",
    "function": {
        "name": "submit_prediction",
        "description": f"Submit your final probability. Must be a number in [{P_MIN}, {P_MAX}].",
        "parameters": {
            "type": "object",
            "properties": {
                "probability": {"type": "number", "description": "P(behavior) for a fresh run in the counterfactual arm"},
                "rationale": {"type": "string", "description": "one or two sentences: the main reason for this number"},
            },
            "required": ["probability", "rationale"],
        },
    },
}

# 12 retries' backoff outlasts OpenRouter's per-minute 429s (Gemini 3.8 Flash is capped at 300 rpm across all its users)
client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], max_retries=12)


def check(arguments: str) -> tuple[dict | None, str]:
    """Parsed submission if valid, else None and the error to send back to the predictor."""
    try:
        args = json.loads(arguments)
        p = float(args["probability"])
    except (ValueError, KeyError, TypeError) as e:
        return None, f"Invalid submission ({e!r}). Call submit_prediction with a numeric probability."
    if not P_MIN <= p <= P_MAX:
        return None, f"probability must be in [{P_MIN}, {P_MAX}], got {p}. Call submit_prediction again."
    return {"p": p, "rationale": str(args.get("rationale", ""))}, "Prediction recorded."


async def predict(predictor: str, prompt: str, tools: dict[str, Tool] | None = None, budget: int = 0, effort: str = EFFORT) -> dict:
    tools = tools or {}
    messages: list = [  # plain dicts in and out; the SDK's TypedDict unions add nothing here
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]
    schemas = [SUBMIT, *(schema for schema, _ in tools.values())]
    cost, used, submission = 0.0, Counter(), None
    for _ in range(budget + MAX_TURNS):
        r = await client.chat.completions.create(
            model=predictor,
            messages=messages,
            tools=schemas,
            max_tokens=16000,
            extra_body={"usage": {"include": True}, "reasoning": {"effort": effort}},
        )
        cost += getattr(r.usage, "cost", 0) or 0
        msg = r.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            messages.append({"role": "user", "content": "Now call submit_prediction with your probability."})
            continue
        for tc in msg.tool_calls:  # every call needs a reply, even after a valid submission
            assert isinstance(tc, ChatCompletionMessageFunctionToolCall)  # we only offer function tools
            name = tc.function.name
            if name == "submit_prediction":
                result, reply = check(tc.function.arguments)
                submission = submission or result
            elif sum(used.values()) >= budget:
                reply = f"Tool budget ({budget} calls) used up. Call submit_prediction now."
            else:
                used[name] += 1
                reply = call(tools, name, tc.function.arguments)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": reply})
        if submission:
            break
    log = {"turns": len(messages), "tool_calls": dict(used), "cost": cost, "tools": schemas, "messages": messages}
    return {**submission, "valid": True, **log} if submission else {"p": None, "rationale": "", "valid": False, **log}
