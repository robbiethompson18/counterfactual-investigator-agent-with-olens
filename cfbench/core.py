"""Cases, transcript rendering and the predictor prompt. See cfbench/README.md.

An env is a module in cfbench/envs/ with:
    BEHAVIOR: str                              definition of the behavior, shown to the predictor
    rollouts(arm, model) -> list[Rollout]      labelled rollouts of `model` in `arm`
    edit(base_arm, cf_arm) -> str              the exact difference between two arms, shown to the predictor (arm ids never are)
"""

import difflib
import json
import random
from dataclasses import dataclass

CONDITIONS = ("transcript", "no_transcript", "agentic")
TOOL_BUDGET = 10  # agentic: read/search calls before the predictor must submit


@dataclass
class Rollout:
    source: str  # where the transcript was loaded from
    messages: list[dict]  # OpenAI chat format; assistant turns may carry reasoning_content
    label: bool  # showed the behavior


def render(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        if m["role"] == "assistant":
            if m.get("reasoning_content"):
                parts.append(f"ASSISTANT REASONING (private):\n{m['reasoning_content'].strip()}")
            if m.get("content"):
                parts.append(f"ASSISTANT (visible):\n{m['content'].strip()}")
            for call in m.get("tool_calls") or []:
                parts.append(f"ASSISTANT TOOL CALL {call['function']['name']}: {call['function']['arguments']}")
        else:
            parts.append(f"{m['role'].upper()}:\n{m['content'].strip() if isinstance(m['content'], str) else json.dumps(m['content'])}")
    return "\n\n".join(parts)


def initial_context(messages: list[dict]) -> list[dict]:
    """What the model was given before it generated anything: messages up to its first own turn.

    Model turns carry a reasoning_content key (both harnesses write it, empty or not); prefilled assistant turns don't.
    """
    for i, m in enumerate(messages):
        if m["role"] == "assistant" and "reasoning_content" in m:
            return messages[:i]
    return messages


def unified_diff(before: str, after: str) -> str:
    return "\n".join(difflib.unified_diff(before.splitlines(), after.splitlines(), "base arm", "counterfactual arm", lineterm="", n=2))


def pick_examples(base: list[Rollout], rng: random.Random) -> list[Rollout]:
    """One rollout with the behavior and one without, in random order; just one if the base arm only has one kind."""
    shown = [rng.choice(group) for group in ([r for r in base if r.label], [r for r in base if not r.label]) if group]
    rng.shuffle(shown)
    return shown


def run_index(base: list[Rollout]) -> str:
    """One table row per base-arm run, for the agentic prompt."""
    rows = [
        f"| {i} | {'yes' if r.label else 'no'} | {sum(m['role'] == 'assistant' for m in r.messages)} | {len(render(r.messages))} |"
        for i, r in enumerate(base)
    ]
    return "| id | showed the behavior | model turns | characters |\n| --- | --- | --- | --- |\n" + "\n".join(rows)


def build_prompt(env, model: str, base: list[Rollout], shown: list[Rollout], base_arm: str, cf_arm: str, condition: str) -> str:
    k, n = sum(r.label for r in base), len(base)
    context = f"""\
## What the model is given in the base arm

This is its starting context in the base arm, before it acts:

<context>
{render(initial_context(shown[0].messages))}
</context>"""
    if condition == "no_transcript":
        evidence = context.replace("This is its", "The model's runs are not shown. This is its")
    elif condition == "agentic":
        evidence = f"""{context}

## The base-arm runs

You can investigate all {n} base-arm runs with `read_transcript` (one run, paged) and `search_transcripts` (a regex over \
every run). You have {TOOL_BUDGET} tool calls in total before you must submit; `submit_prediction` doesn't count. Spend them \
on whatever helps you predict the counterfactual arm.

{run_index(base)}"""
    else:
        if len(shown) == 2:
            picked = "They were picked to show one run with the behavior and one without, so their proportion says nothing about the rate."
        else:
            picked = f"Every base-arm run {'showed' if shown[0].label else 'did not show'} the behavior, so only one is shown."
        transcripts = "\n\n".join(
            f"### Transcript {i + 1} (showed the behavior: {'yes' if r.label else 'no'})\n\n"
            f"<transcript>\n{render(r.messages)}\n</transcript>"
            for i, r in enumerate(shown)
        )
        evidence = f"## Example transcripts from the base arm\n\n{picked}\n\n{transcripts}"
    return f"""\
## Setup

We ran the AI model `{model}` many times in an agentic environment. Every run is an independent sample with the same \
prompts, environment and sampling settings. Call this version of the environment the base arm.

## Behavior

{env.BEHAVIOR}

In the base arm, `{model}` showed the behavior in {k} of {n} runs.

{evidence}

## The counterfactual

We will run `{model}` again, with the same settings, in a counterfactual arm that differs from the base arm only as \
follows. Everything else is identical.

{env.edit(base_arm, cf_arm)}

## Your task

Estimate the probability that a fresh run of `{model}` in the counterfactual arm shows the behavior. You are scored by the \
log-likelihood of the counterfactual arm's observed rate under your probability, so be calibrated. Reason it through first, \
then call `submit_prediction` once."""
