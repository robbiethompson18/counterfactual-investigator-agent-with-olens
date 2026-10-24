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
ANCHOR_CONDITIONS = ("anchors", "anchors_placebo")  # cfbench/anchors.py
TOOL_CONDITIONS = ("agentic", *OLENS_CONDITIONS, *ANCHOR_CONDITIONS)
CONDITIONS = ("transcript", "no_transcript", *TOOL_CONDITIONS)
TOOL_BUDGET = 20  # tool conditions: read/search calls before the predictor must submit
# Bump when the predictor's prompt, tools or scoring change, and don't compare skill across versions. Written to every
# prediction row and meta.json; runs without it are version 1. 2: search_olens over="bullets" (2026-10-24). 3: the tool
# prompts recommend reading one full transcript before searching (2026-10-24). 4: olens tool descriptions no longer
# promise FVE order; reads without FVE get the OLENS_SAMPLED prompt; both O-lens prompts end "Use the O-lens tools
# liberally." (2026-10-24). 5: search_olens returns 20 bullets by default, not 10 (2026-10-24). 6: O-lens hits are
# labelled <transcript>/<olens>, and search hits' context stops at the edges of the read's message field (2026-10-24).
# 7: `--base-runs N` (tools see N random base runs; the prompt says which) (2026-10-24).
HARNESS_VERSION = 7


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
words. Every hit shows the run's own text in <transcript> (⟨READ⟩ marks the token the lens read) and the lens's output for \
that token in <olens>. It was run on the target model's activations at punctuation tokens and turn ends, everywhere in the run (system \
prompt, user turns, tool outputs, its reasoning and replies). Each read is a few short bullets about what the model is \
representing at that point, at one layer, each with its FVE (fraction of the activation's variance the bullet \
explains): high FVE is good, low FVE bullets are often hallucinated. {fve_note} By default the tools show only each \
read's best bullet, and only if its FVE is at least 0.1 (`top_k` and `min_fve` change that); `search_olens` returns the 20 \
highest-FVE bullets (`max_bullets`). `search_olens` with `over="bullets"` searches the bullets' own text instead (every \
bullet, not just the best one) and returns only the bullets that match. Reads can show what the \
model registers but never says; they also often just paraphrase the nearby text. Use the O-lens tools liberally."""

ANCHORS = """

{n_read} of the runs (marked in the table) also have attention reads from the target model's own forward pass over \
that run, sentence by sentence (system prompt, user turns, tool outputs, its reasoning and replies), which you can \
investigate with `anchor_ranking` (a run's top sentences), `anchor_search` (a regex over those runs' sentences) and \
`anchor_influence` (one sentence's effect on later ones). Each sentence gets two scores:
- receiver score: how much attention the model's later reasoning and replies pay back to this sentence, in the attention \
heads that concentrate on a few sentences. High-receiver sentences are the ones the model keeps returning to (plans, \
decisions, key facts).
- suppression effect: cut every path from this sentence to later positions inside the model and measure how much the \
model's predictions of its own later tokens change (mean KL, nats per token). High means what the model wrote \
afterwards depended on this sentence.
Both measure how much the model relies on a sentence in the base arm, not which way a change to it would push the \
behavior. Scores are compared within a run as percentiles (100 = that run's top sentence)."""

# reads from a lens with greedy + sampled readouts and no FVE, at a few chosen positions (scripts/celeste_olens_reads.py)
OLENS_SAMPLED = """

{n_read} of the runs (marked in the table) also have O-lens reads, which you can investigate with `search_olens` (a \
regex over those runs' transcripts that returns the reads at each match) and `olens_at` (all reads around a run and \
character, from any search hit). The O-lens is a model trained to describe another model's internal activations in \
words. Every hit shows the run's own text in <transcript> (⟨READ⟩ marks the token the lens read) and the lens's output for \
that token in <olens>. It was run on the target model's activations at one layer, at about a dozen punctuation tokens per run: the end of \
the final user turn, just before and after the span where this arm's prompt differs from the env's other arms, the final \
reasoning and reply, just before tool calls, and a few spread elsewhere. Each read is the lens's greedy readout followed by \
4 samples, one short bullet each; by default the tools show only the greedy one (`top_k` up to 5 adds the samples, and how \
much they agree is a rough confidence signal). There is no per-read reliability score. Reads can show what the model \
registers but never says; they also often paraphrase or continue the nearby text, and sometimes invent specifics (names, \
numbers, settings) that aren't there. Use the O-lens tools liberally."""


def unified_diff(before: str, after: str) -> str:
    return "\n".join(difflib.unified_diff(before.splitlines(), after.splitlines(), "base arm", "counterfactual arm", lineterm="", n=2))


def pick_examples(base: list[Rollout], rng: random.Random) -> list[Rollout]:
    """One rollout with the behavior and one without, in random order; just one if the base arm only has one kind."""
    shown = [rng.choice(group) for group in ([r for r in base if r.label], [r for r in base if not r.label]) if group]
    rng.shuffle(shown)
    return shown


def run_index(base: list[Rollout], read: set[int] | None = None, label: str = "O-lens reads") -> str:
    """One table row per base-arm run, for the tool prompts; `read` = ids with reads (O-lens or anchor), adds that column."""
    rows = [
        f"| {i} | {'yes' if r.label else 'no'} | {sum(m['role'] == 'assistant' for m in r.messages)} | {len(render(r.messages))} |"
        + (f" {'yes' if i in read else 'no'} |" if read is not None else "")
        for i, r in enumerate(base)
    ]
    head = "| id | showed the behavior | model turns | characters |" + (f" {label} |" if read is not None else "")
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
    fve_note: str | None = "",
    budget: int = TOOL_BUDGET,
    arm: list[Rollout] | None = None,
) -> str:
    """For the olens and anchors conditions: `read` = ids of base runs with reads; olens: `fve_note` = cfbench.olens.fve_note
    (None: the reads have no FVE). `arm`: the whole base arm, when the tools get only some of its runs (`base`)."""
    arm = arm or base
    k, n = sum(r.label for r in arm), len(arm)
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
        if condition in OLENS_CONDITIONS:
            template = OLENS if fve_note is not None else OLENS_SAMPLED
            olens, label = template.format(n_read=len(read or ()), fve_note=fve_note), "O-lens reads"
        elif condition in ANCHOR_CONDITIONS:
            olens, label = ANCHORS.format(n_read=len(read or ())), "anchor reads"
        else:
            olens, label = "", ""
        if len(base) < n:
            runs = f"{len(base)} of the {n} base-arm runs, picked uniformly at random (whatever they showed),"
        else:
            runs = f"all {n} base-arm runs"
        evidence = f"""{context}

## The base-arm runs

You can investigate {runs} with `read_transcript` (one run, paged) and `search_transcripts` (a regex over \
every run). We recommend reading one full transcript before searching them.{olens} You have {budget} tool calls \
in total before you must submit; `submit_prediction` doesn't count. Spend them on whatever helps you predict the counterfactual arm.

{run_index(base, None if condition == "agentic" else read, label)}"""
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

{edit}

## Your task

Estimate the probability that a fresh run of `{model}` in the counterfactual arm shows the behavior. You are scored by the \
log-likelihood of the counterfactual arm's observed rate under your probability, so be calibrated. Reason it through first, \
then call `submit_prediction` once."""
