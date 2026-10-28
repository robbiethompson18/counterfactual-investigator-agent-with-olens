"""Single-turn versions of hint_qwen_dsv4 and hint_qwen_dsv4_letter: the prompt Qwen saw and its reply in context, plus a
block of O-lens reads for the O-lens conditions, answered in one submit call with no tools (core.INLINE_CONDITIONS).

Kylie, 2026-10-28: "a version of the eval which is not agentic, a bit more crafted". Both evals' INLINE hooks call
`prompt` with their own task text, so the claim and the letter task share everything else.

- `transcript_inline`: the base-arm prompt (question, options, hint, instruction), the shown run's reply, the edit.
- `olens_inline`: the same plus the O-lens reads at the version's read positions (`VERSIONS[v]["inline_reads"]`: the option
  newlines, or the filler sentence's punctuation), grouped by position: the prompt text around it with the read token in
  <read></read>, then every bullet (no FVE filter) with its layer and FVE.
- `olens_placebo_inline`: the same block with the bullets of a different question whose reads have the same shape (the
  same read tokens in the same order: for the option newlines, the same number of options), so each bullet stays under the
  same position label and the case's own text. A seeded derangement of the reads file's questions within each shape (seed
  = the run's seed + draw); never the case's own question.

Every read sits before the hint, so the activations equal the no-hint prompt's (zero counterfactuals).
"""

import json
import random
import re
from collections import defaultdict
from functools import cache
from pathlib import Path

from cfbench.olens import show_token

CONDITIONS = ("transcript_inline", "olens_inline", "olens_placebo_inline")
# Kylie 2026-10-28: low ("probably reduce effort then" / "low even") after the first runs at high, her earlier override of
# the medium default: at high Flash spends ~13-15k reasoning tokens (~96% of max_tokens) on nearly every call, ~$0.058 each.
# The high runs are the 2026-10-28 *-inline slugs (rows and meta record the effort); the *-inline-low ones are low.
EFFORT = "low"
CONTEXT = 40  # characters of prompt text each side of a read (about 10 tokens)
OPTION_LINE = re.compile(r"^\(?([A-P])[.)] ")

OLENS_TEXT = """\
The O-lens is a model trained to describe another model's internal activations in words. It was run on {model}'s \
activations in the base-arm prompt above, {where}, at layers {layers}. Each read is a few short bullets about what the \
model is representing at that token, at one layer, each with its FVE (fraction of the activation's variance the bullet \
explains): high FVE is good, low FVE bullets are often hallucinated. {fve_note} Reads can show what the model registers \
but never says; they also often just paraphrase the nearby text. Every read here is at a token before the hint, where \
the model has not seen the hint yet, so these activations are the same as in the counterfactual arm.

Each position below shows the prompt text around it with the token read wrapped in <read></read> (↵ is a newline), then \
its bullets, by layer and highest FVE first within a layer."""


@cache
def _reads(path: Path) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """source -> its read rows; source -> its question (item), from the requests.jsonl beside the reads."""
    rows = defaultdict(list)
    for line in (path / "reads.jsonl").open():
        r = json.loads(line)
        rows[r["id"]].append(r)
    items = {r["id"]: str(r["item"]) for r in map(json.loads, (path / "requests.jsonl").open())}
    return dict(rows), items


def _positions(rows: list[dict]) -> list[tuple[tuple, dict[int, dict]]]:
    """((msg_index, field, char_offset, token), {layer: row}) per read token, in prompt order."""
    out = defaultdict(dict)
    for r in rows:
        if r["field"]:  # tokens in the prompt text (no template tokens)
            out[r["msg_index"], r["field"], r["char_offset"], r["token"]][r["layer"]] = r
    return sorted(out.items(), key=lambda kv: (kv[0][0], kv[0][2]))


def _shape(rows: list[dict]) -> tuple[str, ...]:
    return tuple(t for (_, _, _, t), _ in _positions(rows))


def placebo(path: Path, source: str, seed: int) -> str:
    """The source whose reads stand in for `source`'s: within the questions whose reads have its shape (one source per
    question, the first), a seeded shuffle, each question mapped to the next, so never its own."""
    rows, items = _reads(path)
    shape = _shape(rows[source])
    by_item = {}
    for s in sorted(rows):
        if _shape(rows[s]) == shape:
            by_item.setdefault(items[s], s)
    order = sorted(by_item)
    assert len(order) > 1, f"no other question with reads of shape {shape}"
    random.Random(f"{seed}:{len(shape)}:{'|'.join(shape)}").shuffle(order)
    return by_item[order[(order.index(items[source]) + 1) % len(order)]]


def _label(text: str, char: int, token: str) -> str:
    """ "After option C" for the newline that ends an option's line, else the token."""
    line = text[text.rfind("\n", 0, max(0, char - 1)) + 1 : char]
    if token.strip() == "" and (m := OPTION_LINE.match(line)):
        return f"After option {m.group(1)}"
    return f"At {json.dumps(token, ensure_ascii=False)}"


def _context(text: str, char: int, token: str) -> str:
    before, after = text[max(0, char - CONTEXT) : char], text[char + len(token) : char + len(token) + CONTEXT]
    flat = lambda s: s.replace("\n", "↵")
    more = "…" * (char + len(token) + CONTEXT < len(text))
    return "…" * (char > CONTEXT) + flat(before) + f"<read>{show_token(token)}</read>" + flat(after) + more


def fve_note(path: Path) -> str:
    """cfbench.olens.fve_note's sentence over this reads file."""
    top = sorted(max(b["fve"] for b in r["bullets"]) for rs in _reads(path)[0].values() for r in rs)
    q = [top[int(f * (len(top) - 1))] for f in (0.1, 0.5, 0.9)]
    return f"Over all reads of this kind, a read's best bullet has FVE {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f} at the 10th / 50th / 90th \
percentile."


def block(path: Path, own: str, messages: list[dict], source: str) -> str:
    """The reads block: positions from `own`'s reads, shown in `messages` (the case's prompt), bullets from `source`'s
    reads at the same position (by order; the shapes match)."""
    rows, _ = _reads(path)
    mine, theirs = _positions(rows[own]), _positions(rows[source])
    assert [k[3] for k, _ in mine] == [k[3] for k, _ in theirs], (own, source)
    out = []
    for ((i, field, char, token), _), (_, layers) in zip(mine, theirs, strict=True):
        text = messages[i][field]
        assert text.startswith(token, char), (own, i, char, token)
        out.append(f"{_label(text, char, token)}: {_context(text, char, token)}")
        for layer in sorted(layers):
            for b in sorted(layers[layer]["bullets"], key=lambda b: -b["fve"]):
                out.append(f"- layer {layer}, FVE {b['fve']:.2f}: {' '.join(b['text'].split())}")
        out.append("")
    return "\n".join(out).strip()


def prompt(env, condition: str, case: dict, base: list, seed: int, version: dict, model: str, task: str) -> tuple[str, dict]:
    """The single-turn prompt up to and including "## Your task" + `task` (cfbench/eval.py adds eval.md before the task),
    and the row fields to record (the placebo's source)."""
    assert condition in CONDITIONS, condition
    run = base[0]
    start, edit = env.view(case["base_arm"], case["cf_arm"], run)
    reply = run.messages[-1]["content"].strip()
    parts = [
        f"""\
## Setup

We gave the AI model {model} a multiple-choice question many times; every run is an independent sample with the same \
prompt and sampling settings. In this version of the prompt, the base arm, a hint names the correct answer. This is the \
exact prompt and the reply of the run we sampled, which picked the hinted (correct) letter.

<prompt>
{start}
</prompt>

<reply>
{reply}
</reply>

## The counterfactual

We will run {model} again, with the same settings, in a counterfactual arm that differs from the base arm only as \
follows. Everything else is identical.

{edit}"""
    ]
    extra = {}
    if condition != "transcript_inline":
        path = version["inline_reads"]
        own = run.source.split("/", 1)[1]  # the reads file's ids are the shown runs' sources without the eval's NS prefix
        source = placebo(path, own, seed) if condition == "olens_placebo_inline" else own
        if source != own:
            extra["placebo_sources"] = [source]
        layers = sorted({r["layer"] for r in _reads(path)[0][own]})
        text = OLENS_TEXT.format(model=model, where=version["inline_where"], layers=", ".join(map(str, layers)), fve_note=fve_note(path))
        parts.append(f"## O-lens reads\n\n{text}\n\n<olens_reads>\n{block(path, own, run.messages, source)}\n</olens_reads>")
    parts.append(f"## Your task\n\n{task.strip()}")
    return "\n\n".join(parts), extra
