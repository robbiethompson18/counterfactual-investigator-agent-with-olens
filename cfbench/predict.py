"""The predictor: one conversation that ends when it submits a valid probability through the submit_prediction tool."""

import json
import os

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionFunctionToolParam, ChatCompletionMessageFunctionToolCall

P_MIN, P_MAX = 0.01, 0.99
MAX_TURNS = 4  # a predictor that can't submit a valid number in 4 turns is recorded as invalid

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

client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], max_retries=5)


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


async def predict(predictor: str, prompt: str) -> dict:
    messages: list = [  # plain dicts in and out; the SDK's TypedDict unions add nothing here
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompt},
    ]
    cost = 0.0
    for _ in range(MAX_TURNS):
        r = await client.chat.completions.create(
            model=predictor, messages=messages, tools=[SUBMIT], max_tokens=16000, extra_body={"usage": {"include": True}}
        )
        cost += getattr(r.usage, "cost", 0) or 0
        msg = r.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            messages.append({"role": "user", "content": "Now call submit_prediction with your probability."})
            continue
        call = msg.tool_calls[0]
        assert isinstance(call, ChatCompletionMessageFunctionToolCall)  # we only offer function tools
        submission, reply = check(call.function.arguments)
        messages.append({"role": "tool", "tool_call_id": call.id, "content": reply})
        if submission:
            return {**submission, "valid": True, "turns": len(messages), "cost": cost, "messages": messages}
    return {"p": None, "rationale": "", "valid": False, "turns": len(messages), "cost": cost, "messages": messages}
