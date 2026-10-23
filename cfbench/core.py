"""Cases, transcript rendering and the predictor prompt. See cfbench/README.md.

An env is a module in cfbench/envs/ with:
    BEHAVIOR: str                              definition of the behavior, shown to the predictor
    rollouts(arm, model) -> list[Rollout]      labelled rollouts of `model` in `arm`
    edit(base_arm, cf_arm) -> str              the exact difference between two arms, shown to the predictor (arm ids never are)
or, for envs whose arms run over several task variants, in place of edit (and the optional context(rollout)):
    view(base_arm, cf_arm, rollout) -> (starting context, edit), both on one variant
"""

import difflib
import json
import random
from dataclasses import dataclass

OLENS_CONDITIONS = ("olens", "olens_placebo")  # cfbench/olens.py
TOOL_CONDITIONS = ("agentic", *OLENS_CONDITIONS)
# `transcript` with reads written into the transcripts; the _cf ones also show the counterfactual arm's starting context with
# its reads (a forward pass over the edited prompt, no generation)
INLINE_CONDITIONS = ("transcript_olens", "transcript_olens_placebo", "transcript_olens_cf", "transcript_olens_cf_placebo")
CONDITIONS = ("transcript", "no_transcript", *TOOL_CONDITIONS, *INLINE_CONDITIONS)
TOOL_BUDGET = 20  # tool conditions: read/search calls before the predictor must submit


@dataclass
class Rollout:
    source: str  # where the transcript was loaded from
    messages: list[dict]  # OpenAI chat format; assistant turns may carry reasoning_content
    label: bool  # showed the behavior


def render_spans(messages: list[dict]) -> tuple[str, dict[tuple[int, str], int]]:
    """render() plus where each message field sits in it: (message index, field) -> b such that the field's raw character c
    is at text[b + c] ("tool_call" -> the start of that message's first tool call line)."""
    parts, spans, at = [], {}, 0

    def add(header: str, key: tuple[int, str], raw: str | None = None):
        nonlocal at
        spans[key] = at + len(header) - (len(raw) - len(raw.lstrip()) if raw is not None else 0)
        parts.append(header + (raw.strip() if raw is not None else ""))
        at += len(parts[-1]) + 2  # the "\n\n" join

    for i, m in enumerate(messages):
        if m["role"] == "assistant":
            if m.get("reasoning_content"):
                add("ASSISTANT REASONING (private):\n", (i, "reasoning_content"), m["reasoning_content"])
            if m.get("content"):
                add("ASSISTANT (visible):\n", (i, "content"), m["content"])
            for j, call in enumerate(m.get("tool_calls") or []):
                add(f"ASSISTANT TOOL CALL {call['function']['name']}: {call['function']['arguments']}", (i, f"tool_call{j or ''}"))
        elif isinstance(m["content"], str):
            add(f"{m['role'].upper()}:\n", (i, "content"), m["content"])
        else:
            add(f"{m['role'].upper()}:\n{json.dumps(m['content'])}", (i, "content"))
    return "\n\n".join(parts), spans


def render(messages: list[dict]) -> str:
    return render_spans(messages)[0]


def initial_context(messages: list[dict]) -> list[dict]:
    """What the model was given before it generated anything: messages up to its first own turn.

    Model turns carry a reasoning_content key (both harnesses write it, empty or not); prefilled assistant turns don't.
    """
    for i, m in enumerate(messages):
        if m["role"] == "assistant" and "reasoning_content" in m:
            return messages[:i]
    return messages


OLENS = """

{n_read} of the runs (marked in the table) also have O-lens reads, which you can investigate with `search_olens` (a \
regex over those runs' transcripts that returns the reads at each match) and `olens_at` (all reads around a run and \
character, from any search hit). The O-lens is a model trained to describe another model's internal activations in \
words. It was run on the target model's activations at punctuation tokens and turn ends, everywhere in the run (system \
prompt, user turns, tool outputs, its reasoning and replies). Each read is a few short bullets about what the model is \
representing at that point, at one layer, each with its FVE (fraction of the activation's variance the bullet \
explains): high FVE is good, low FVE bullets are often hallucinated. {fve_note} By default the tools show only each \
read's best bullet, and only if its FVE is at least 0.1 (`top_k` and `min_fve` change that); `search_olens` returns the 10 \
highest-FVE bullets (`max_bullets`). Reads can show what the \
model registers but never says; they also often just paraphrase the nearby text."""

INLINE = """\
The transcripts below also carry O-lens reads, written inline as ⟨O-lens: ...⟩ right after the token they were read at. \
The O-lens is a model trained to describe another model's internal activations in words. It was run on the target \
model's activations at punctuation tokens and turn ends, everywhere in the run (system prompt, user turns, its reasoning \
and replies). Each read gives, per layer (L32, L40, L48), the best bullet about what the model is representing at that \
point, with its FVE (fraction of the activation's variance the bullet explains): high FVE is good, low FVE bullets are \
often hallucinated; bullets under FVE 0.1 are left out. {fve_note} Reads can show what the model registers but never \
says; they also often just paraphrase the nearby text."""


CF_READS = """
We also ran `{model}` forward over the counterfactual arm's starting context, without generating anything (so this is not \
a run of the counterfactual arm and says nothing about what it does next), and read those activations with the O-lens, \
in the same format as above:

<counterfactual_context>
{cf_render}
</counterfactual_context>
"""


def unified_diff(before: str, after: str) -> str:
    return "\n".join(difflib.unified_diff(before.splitlines(), after.splitlines(), "base arm", "counterfactual arm", lineterm="", n=2))


def pick_examples(base: list[Rollout], rng: random.Random) -> list[Rollout]:
    """One rollout with the behavior and one without, in random order; just one if the base arm only has one kind."""
    shown = [rng.choice(group) for group in ([r for r in base if r.label], [r for r in base if not r.label]) if group]
    rng.shuffle(shown)
    return shown


def run_index(base: list[Rollout], read: set[int] | None = None) -> str:
    """One table row per base-arm run, for the tool prompts; `read` = ids with O-lens reads, adds that column."""
    rows = [
        f"| {i} | {'yes' if r.label else 'no'} | {sum(m['role'] == 'assistant' for m in r.messages)} | {len(render(r.messages))} |"
        + (f" {'yes' if i in read else 'no'} |" if read is not None else "")
        for i, r in enumerate(base)
    ]
    head = "| id | showed the behavior | model turns | characters |" + (" O-lens reads |" if read is not None else "")
    return head + "\n" + "| --- " * (head.count("|") - 1) + "|\n" + "\n".join(rows)


def build_prompt(
    env,
    model: str,
    base: list[Rollout],
    shown: list[Rollout],
    base_arm: str,
    cf_arm: str,
    condition: str,
    read: set[int] | None = None,
    fve_note: str = "",
    renders: dict[str, str] | None = None,
    cf_render: str | None = None,
) -> str:
    """For the olens conditions: `read` = ids of base runs with O-lens reads, `fve_note` = cfbench.olens.fve_note.
    For the inline conditions: `renders` = each shown run's source -> its transcript with reads (cfbench.olens.annotate),
    `cf_render` = the counterfactual arm's starting context with its reads (the _cf conditions)."""
    k, n = sum(r.label for r in base), len(base)
    if hasattr(env, "view"):  # multi-variant envs: starting context and edit from the same variant
        start, edit = env.view(base_arm, cf_arm, shown[0])
    else:
        start = env.context(shown[0]) if hasattr(env, "context") else render(initial_context(shown[0].messages))
        edit = env.edit(base_arm, cf_arm)
    context = f"""\
## What the model is given in the base arm

This is its starting context in the base arm, before it acts:

<context>
{start}
</context>"""
    if condition == "no_transcript":
        evidence = context.replace("This is its", "The model's runs are not shown. This is its")
    elif condition in TOOL_CONDITIONS:
        olens = "" if condition == "agentic" else OLENS.format(n_read=len(read or ()), fve_note=fve_note)
        evidence = f"""{context}

## The base-arm runs

You can investigate all {n} base-arm runs with `read_transcript` (one run, paged) and `search_transcripts` (a regex over \
every run).{olens} You have {TOOL_BUDGET} tool calls in total before you must submit; `submit_prediction` doesn't count. \
Spend them on whatever helps you predict the counterfactual arm.

{run_index(base, None if condition == "agentic" else read)}"""
    else:
        if len(shown) == 2:
            picked = "They were picked to show one run with the behavior and one without, so their proportion says nothing about the rate."
        else:
            picked = f"Every base-arm run {'showed' if shown[0].label else 'did not show'} the behavior, so only one is shown."
        transcripts = "\n\n".join(
            f"### Transcript {i + 1} (showed the behavior: {'yes' if r.label else 'no'})\n\n"
            f"<transcript>\n{renders[r.source] if renders else render(r.messages)}\n</transcript>"
            for i, r in enumerate(shown)
        )
        inline = f"\n\n{INLINE.format(fve_note=fve_note)}" if renders else ""
        evidence = f"## Example transcripts from the base arm\n\n{picked}{inline}\n\n{transcripts}"
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

{edit}
{CF_READS.format(model=model, cf_render=cf_render) if cf_render else ""}
## Your task

Estimate the probability that a fresh run of `{model}` in the counterfactual arm shows the behavior. You are scored by the \
log-likelihood of the counterfactual arm's observed rate under your probability, so be calibrated. Reason it through first, \
then call `submit_prediction` once."""
