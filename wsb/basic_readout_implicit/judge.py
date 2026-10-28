"""WorkspaceBench's single-token bank judge, copied verbatim so a text reader is graded by the same instrument as the lens.

Source: https://github.com/camilablank/workspace-bench at 86612fa4925100391b39c27894807a479595c61c (MIT, Camila Blank):
`src/wsbench/basic/prompts.py` (SYSTEM, KIND_NOTE_PROSE, USER, BLOCK, SCHEMA, PROMPT_VERSION) and `src/wsbench/basic/judge.py`
(`_fold`, `_in_sample`, `quote_verified`). basic_readout has no family rule and no forbidden-answer line, so those are omitted.
Their judge runs `google/gemini-3.8-flash` at reasoning effort `minimal` (`judge_config.py`), temperature 0, strict JSON schema.
Never import or run workspace-bench; copy what you need."""

import json
import re
import unicodedata

PROMPT_VERSION = "bank-2026-10-10"

SYSTEM = """You are judging readouts from an activation lens for an interpretability
benchmark. You get one eval item: its TARGET concept(s) and the lens's readout samples at one
layer (several token positions; for the J-lens the samples are top-10 vocabulary tokens, for
the oracle lens they are free-text generations).

Decide whether ANY target concept is EXPRESSED in the readout. Grade meaning, not wording:
- Expressed means a sample states the target's MEANING: the word itself (any case or
  inflection), an established synonym or alias, a faithful translation into any language, a
  paraphrase, or a description that states or depicts the target itself (target "let the
  executive die" is expressed by "ignored the freezing man, leading to his demise").
- A narrower instance that necessarily entails the target counts ("a puppy" expresses "dog";
  "the toddler years" expresses "child").
- Several phrases inside ONE sample may together express the target when between them they
  state its specific content ("off-by-one | fencepost | hotel nights" expresses "fencepost
  error in the nights count"). Pieces in DIFFERENT samples or different tokens NEVER combine.
- NOT expressed by clues, circumstances or evidence from which the target could be inferred,
  including a restated question, index, definition or calculation: "the element with atomic
  number 26" does NOT express "iron"; "six divided by two" does NOT express "three"; "crayon
  drawings on the fridge" does NOT express "child". The sample must state the target itself.
- NOT expressed by a broader category (hypernym), a merely related or co-occurring idea, or the
  general topic.
- NOT expressed by a different or derived word that merely contains the target string
  ("blackout" does NOT express "black"), or by word fragments or partial tokens ("paragus"
  does NOT express "asparagus").
- Match the direction of the claim: if the target asserts a stance, a sample asserting the
  opposite, or naming the topic without the stance, does NOT express it.
- A sample that lists many interchangeable candidates of the same type (most of the digits,
  a run of month or sport names) expresses none of them unless it singles one out (states it
  as the answer, or sets it apart).
- A family rule or a forbidden-answer line, when given below the targets, overrides the rules
  above.
- Judge only what is actually written. If uncertain, say not expressed.
- Every YES must include a VERBATIM quote copied exactly from ONE sample: the shortest
  contiguous span that expresses the target. No quote, no pass.
Return JSON: {"expressed": bool, "target": "<which target, or empty>", "quote": "<verbatim>"}."""

KIND_NOTE_PROSE = "The samples below are free-text readouts."
USER = "targets: {targets}\n{kind_note}{family_rule}{forbidden}\n\n{blocks}"
BLOCK = "[position {pos}]\n{samples}"

SCHEMA = {
    "name": "bank_verdict",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["expressed", "target", "quote"],
        "properties": {"expressed": {"type": "boolean"}, "target": {"type": "string"}, "quote": {"type": "string"}},
    },
}


def render_user(targets: list[str], samples: list[str], pos: int = 0) -> str:
    """Their `render_user` for one position's free-text samples, no family rule, no forbidden line."""
    block = BLOCK.replace("{pos}", str(pos)).replace("{samples}", "\n".join(f"  {s}" for s in samples))
    return (
        USER.replace("{targets}", json.dumps(targets))
        .replace("{kind_note}", KIND_NOTE_PROSE)
        .replace("{family_rule}", "")
        .replace("{forbidden}", "")
        .replace("{blocks}", block)
    )


_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")


def _fold(s: str) -> str:
    """Lowercase, decode literal ``\\uXXXX`` escapes, strip accents, drop non-alphanumerics."""
    s = _ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), s.lower())
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if c.isalnum() and not unicodedata.combining(c))


def _in_sample(quote: str, sample: str) -> bool:
    if quote in sample:
        return True
    norm_q = " ".join(quote.lower().split())
    if norm_q and norm_q in " ".join(sample.lower().split()):
        return True
    folded_q = _fold(quote)
    return bool(folded_q) and folded_q in _fold(sample)


def quote_verified(quote: str, samples: list[str]) -> bool:
    """A positive's quote must be a span of ONE readout sample: exact, then whitespace- and
    case-normalised, then folded (punctuation, markdown and accents ignored). A quote that only
    exists across two samples never verifies; an empty quote never verifies."""
    if not quote.strip():
        return False
    return any(_in_sample(quote, s) for s in samples)


def verdict(result: dict, samples: list[str]) -> dict:
    """Their `_verdict`: a YES whose quote isn't verbatim in one sample is voided."""
    quote = str(result.get("quote") or "")
    quote_ok = quote_verified(quote, samples)
    return {
        "expressed": bool(result.get("expressed")) and quote_ok,
        "target": str(result.get("target") or ""),
        "quote": quote,
        "quote_ok": quote_ok,
    }
