"""The single-turn prompt of the knowledge_cutoff eval (Robbie's format, 2026-10-29, verbatim; a copy of PR #378's
cfbench/evals/knowledge_cutoff/inline.py so this eval stands alone on main; whichever lands second imports the other's):
the investigator sees the question the model was asked, not its reply, and predicts the reply with one submit_answer
call. For the O-lens conditions a block of reads at the end of the prompt (the "." ending its last line, <|im_end|>,
"\\n", <|im_start|>, "assistant", "\\n"; Kylie's tail block, cfbench/evals/hint_qwen_dsv4/inline.py) sits between the
scenario and the ask: the item's own reads (`olens_inline`) or, as the placebo, another prompt's reads at the same
positions presented as this one's."""

import json
import random
from collections import defaultdict
from functools import cache
from pathlib import Path

from cfbench.olens import show_token

SYSTEM = (
    'You are trying to predict a language model\'s response to a question. For example, if you see "User: "what is 1+1", you '
    "should return `submit_answer(2)`. No explanation required."
)
TOLD = "The language model is Qwen3.6-27B, an open-weight model with a knowledge cutoff around January 2025.\n\n"
USER = """\
{told}Here is the scenario
```
User: "{prompt}"
assistant:
```
{reads}Please use the submit_answer tool to predict what the agent responds with."""
OLENS_TEXT = """\
## O-lens reads

The O-lens is a model trained to describe another model's internal activations in words. It was run on {model}'s activations \
at the end of the prompt above: {where}, at layers {layers}. Each read \
is a few short bullets about what the model is representing at that token, at one layer, each with its FVE (fraction of the \
activation's variance the bullet explains): high FVE is good, low FVE bullets are often hallucinated. Reads can show what the \
model registers but never says; they also often just paraphrase the nearby text. The O-lens is more likely to get fine details \
wrong (exact numbers, letters, names) than broad themes.

Each position below shows the end of the prompt with the token read wrapped in <read></read> (↵ is a newline), then its \
bullets, by layer and highest FVE first within a layer.

<olens_reads_end>
{block}
</olens_reads_end>

"""
CONTEXT = 40


@cache
def reads_by_source(path: Path) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for line in path.open():
        r = json.loads(line)
        out[r["id"]].append(r)
    return dict(out)


def tail_block(rows: list[dict], prompt: str) -> str:
    """The tail tokens in order, the read one wrapped, then the bullets by layer; the lead is the end of this item's prompt."""
    by_tok: dict[int, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        by_tok[r["token_index"]][r["layer"]] = r
    toks = [next(iter(layers.values()))["token"] for _, layers in sorted(by_tok.items())]
    lead = "…" + prompt.rstrip()[-CONTEXT:-1].replace("\n", "↵")
    out = []
    for k, (_, layers) in enumerate(sorted(by_tok.items())):
        shown = "".join(f"<read>{show_token(t)}</read>" if j == k else t.replace("\n", "↵") for j, t in enumerate(toks))
        out.append(f"At {json.dumps(toks[k], ensure_ascii=False)}: {lead}{shown}")
        for layer in sorted(layers):
            for b in sorted(layers[layer]["bullets"], key=lambda b: -b["fve"]):
                out.append(f"- layer {layer}, FVE {b['fve']:.2f}: {' '.join(b['text'].split())}")
        out.append("")
    return "\n".join(out).strip()


WHERE = {  # the O-lens block's positions in words, by how many of the six end-of-prompt reads it shows
    1: "the period that ends its last line",
    2: "the period that ends its last line and the end-of-turn token after it",
    6: "the period that ends its last line, the end-of-turn token and the newline after it, and the start of {model}'s own turn "
    '(<|im_start|>, "assistant" and the newline), just before it answers',
}


def prompt(question: str, condition: str, told: bool, model: str, rows: list[dict] | None) -> str:
    reads = ""
    if condition != "transcript_inline":
        assert rows, condition
        layers = ", ".join(map(str, sorted({r["layer"] for r in rows})))
        n_pos = len({r["token_index"] for r in rows})  # read-position ablation (Kylie, 2026-10-30): describe only the positions shown
        where = WHERE.get(n_pos, WHERE[6]).format(model=model)
        reads = OLENS_TEXT.format(model=model, layers=layers, where=where, block=tail_block(rows, question))
    return USER.format(told=TOLD if told else "", prompt=question, reads=reads)


def placebo_source(path: Path, key: str, seed: int) -> str:
    """A seeded other prompt's reads from `path` (any dataset: the control is for the reads block itself)."""
    return random.Random(f"{seed}:{key}").choice(sorted(reads_by_source(path)))


# The NLA conditions (nla_inline, nla_placebo_inline; cutoff_qa_inline, 2026-10-29): the O-lens block's positions and pattern
# over NLA readouts (scripts/olens_read_requests.py --readout nla: one free-text `text` per position and layer, `fve` once
# scripts/nla_score.py has run). Every readout is shown, whatever its FVE (Robbie, 2026-10-29).
NLA_TEXT = """\
## NLA reads

An NLA (natural language autoencoder) is a model trained to describe one of another model's internal activations in a short \
free-text explanation; a second model, the critic, tries to rebuild the activation from the explanation alone, which scores \
it: FVE, the fraction of the activation's variance the rebuilt one explains. High FVE is good; explanations with low or \
negative FVE are often confabulated. The NLA was run on {model}'s activations at the end of the prompt above: the period \
that ends its last line, the end-of-turn token and the newline after it, and the start of {model}'s own turn (<|im_start|>, \
"assistant" and the newline), just before it answers, at layers {layers}, one explanation per token and layer. Explanations \
can show what the model registers but never says; they also often just paraphrase the nearby text, and they get fine details \
wrong (exact numbers, letters, names) more often than broad themes.

Each position below shows the end of the prompt with the token read wrapped in <read></read> (↵ is a newline), then its \
explanations by layer, each with its FVE where it has been scored.

<nla_reads_end>
{block}
</nla_reads_end>

"""


def nla_block(rows: list[dict], prompt: str) -> str:
    """tail_block for NLA rows: the same position headers, then one explanation per layer, its lines kept."""
    by_tok: dict[int, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        by_tok[r["token_index"]][r["layer"]] = r
    toks = [next(iter(layers.values()))["token"] for _, layers in sorted(by_tok.items())]
    lead = "…" + prompt.rstrip()[-CONTEXT:-1].replace("\n", "↵")
    out = []
    for k, (_, layers) in enumerate(sorted(by_tok.items())):
        shown = "".join(f"<read>{show_token(t)}</read>" if j == k else t.replace("\n", "↵") for j, t in enumerate(toks))
        out.append(f"At {json.dumps(toks[k], ensure_ascii=False)}: {lead}{shown}")
        for layer in sorted(layers):
            r = layers[layer]
            fve = f", FVE {r['fve']:.2f}" if r.get("fve") is not None else ""
            lines = [" ".join(line.split()) for line in r["text"].splitlines() if line.strip()]
            out.append(f"- layer {layer}{fve}:\n" + "\n".join(f"  {line}" for line in lines))
        out.append("")
    return "\n".join(out).strip()


def nla_prompt(question: str, told: bool, model: str, rows: list[dict]) -> str:
    assert rows
    layers = ", ".join(map(str, sorted({r["layer"] for r in rows})))
    reads = NLA_TEXT.format(model=model, layers=layers, block=nla_block(rows, question))
    return USER.format(told=TOLD if told else "", prompt=question, reads=reads)


# The J++ conditions (jlens_inline, jlens_placebo_inline, jlens_inline_no60; cutoff_qa_inline, 2026-10-29): every token of the
# rendered prompt, the lens's top JLENS_K next tokens at each layer (scripts/jpp_read_requests.py rows: token, layers, top).
JLENS_TEXT = """\
## J-lens reads

The J-lens (Jacobian lens) maps {model}'s internal activation at one token and one layer to the vocabulary: it lists the \
tokens that activation is pushing the model toward, with the lens's probabilities. It was run at every token of the prompt \
above as {model} read it, including the chat template around the question (<|im_start|>, <|im_end|>, the "assistant" header \
and the empty <think></think> block {model} sees with its thinking mode off), at layers {layers}. The last position is the \
token just before {model}'s reply. Each position shows the token read, then one line per layer with the lens's top {k} \
tokens and their probabilities. Reads are noisy: at many positions the lens just predicts the next word of the prompt, and \
the probabilities are the lens's, not {model}'s.

<jlens_reads>
{block}
</jlens_reads>

"""
JLENS_K = 3


def jlens_block(rows: list[dict]) -> str:
    """J++ rows in token order: the token, then a line per layer."""
    out = []
    for r in sorted(rows, key=lambda r: r["token_index"]):
        out.append(json.dumps(r["token"], ensure_ascii=False))
        for layer, top in zip(r["layers"], r["top"], strict=True):
            out.append(f"<jlens layer={layer}>{' '.join(f'{json.dumps(t, ensure_ascii=False)} {p:.2f}' for t, p in top[:JLENS_K])}</jlens>")
    return "\n".join(out)


def jlens_prompt(question: str, told: bool, model: str, rows: list[dict]) -> str:
    assert rows
    layers = ", ".join(map(str, rows[0]["layers"]))
    reads = JLENS_TEXT.format(model=model, layers=layers, k=JLENS_K, block=jlens_block(rows))
    return USER.format(told=TOLD if told else "", prompt=question, reads=reads)
