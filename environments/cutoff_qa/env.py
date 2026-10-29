"""cutoff_qa: dated short-answer factual questions (events 2024-01 .. 2026-09), each asked as a bare user turn with no system
prompt and thinking off, the way WorkspaceBench's knowledge_cutoff family renders its items (Robbie, 2026-10-29).

The subject (Qwen3.6-27B) has a knowledge cutoff around January 2025, so it tends to answer the 2024 questions correctly and
the later ones with a stale value (last year's winner, the previous CEO). Items are the `cutoff_qa` bank: 1,072 questions
written by 25 Sonnet subagents from Wikipedia year pages and web search, web-verified, with a short canonical `truth`, its
`truth_aliases`, and for 58% a `pre_cutoff_analog` (the older value a Jan-2025 model might give). The bank is data, so it
lives on the HF dataset at `banks/cutoff_qa/v1.json` (fetched on first use); `scripts/cutoff_facts/merge.py` builds it and
docs/envs/cutoff_qa.md says how it was made and what is wrong with it.

Arms: `nohint` only, for now: the user message is the bank's `question` string verbatim (it ends "If you are not sure, give
your best guess. Answer with just the <answer_type>."). TODO: a `hint` arm that prefixes "A Stanford professor said it's
{truth}." (or the stale analog) the way hinted_truthfulqa's `hint/stanford` does, as the counterfactual; not built yet.

Measures (no LLM judge anywhere): `answer` = the reply normalized (`norm`: lowercase, accents and punctuation stripped,
leading article dropped, currency / units / digit-group commas stripped from numbers, whitespace collapsed); `correct` = it
equals the normalized truth or an alias (PRIMARY); `stale` = it equals the normalized pre-cutoff analog (None when the item
has none); `contains_truth` / `contains_stale` = the same as a whole phrase inside the reply, for replies that add words.
"""

import json
import re
import unicodedata
from functools import cache

from counterfactual_investigator_agent_with_olens.runs import REPO
from environments._lib.core import Ctx, Episode
from environments._lib.runfolder import REPO_ROOT

ARMS = ["nohint"]
BANK = REPO_ROOT / "data/banks/cutoff_qa/v1.json"  # HF banks/cutoff_qa/v1.json, fetched on first use


@cache
def bank() -> dict:
    if not BANK.exists():
        from huggingface_hub import hf_hub_download

        hf_hub_download(REPO, "banks/cutoff_qa/v1.json", repo_type="dataset", local_dir=REPO_ROOT / "data")
    return json.loads(BANK.read_text())


@cache
def _items() -> dict[str, dict]:
    return {it["id"]: it for it in bank()["items"]}


def items():
    return list(_items())


def build(arm, item, ctx: Ctx | None = None) -> Episode:
    it = _items()[item]
    meta = {k: it[k] for k in ("truth", "truth_aliases", "pre_cutoff_analog", "event_date", "domain", "answer_type", "numeric")}
    return Episode(messages=[{"role": "user", "content": it["question"]}], meta=meta)


UNITS = (
    r"(?:million|billion|trillion|thousand|crore|lakh|bn|mn|m|k|km|mm|cm|kg|mw|gw"
    r"|usd|us dollars|dollars|euros|pounds|yuan|francs|chf|gbp|eur|percent)"
)
NUMBER_UNITS = re.compile(rf"\b(\d+(?:\.\d+)?)(?:\s*{UNITS}\b)+")


def norm(s: str) -> str:
    """The comparison form of an answer: lowercase ASCII, no quotes or punctuation (decimal points kept), no leading article,
    no currency symbols, digit-group commas or unit words on numbers ("$32.5 million" -> "32.5"), single spaces."""
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[\"'“”‘’`´]", "", s)  # quotes and apostrophes vanish ("harry's" -> "harrys", so "harrys" and "harry's" agree)
    s = re.sub(r"(?<=\d),(?=\d)", "", s)  # 762,400 -> 762400
    s = re.sub(r"[^\w\s.]", " ", s)  # other punctuation (incl. currency symbols, hyphens) -> space
    s = re.sub(r"\.(?!\d)", " ", s)  # dots not starting a decimal -> space ("apt." -> "apt", "4.002" kept)
    s = NUMBER_UNITS.sub(r"\1", s)
    s = re.sub(r"^(?:the|a|an)\s+", "", s.strip())
    return " ".join(s.split())


def analog_key(analog: str | None) -> str | None:
    """The stale value's comparison form: the analog text before any parenthetical ("sonoma (macOS 14, 2023)" -> "sonoma")."""
    return norm(analog.split("(")[0]) or None if analog else None


def _contains(phrase: str, text: str) -> bool:
    return re.search(rf"(?<!\S){re.escape(phrase)}(?!\S)", text) is not None


def measure(ep) -> dict:
    meta = ep.get("meta") or {}
    reply = next((m.get("content") or "" for m in reversed(ep["messages"]) if m["role"] == "assistant"), "")
    answer = norm(reply)
    truths = {norm(t) for t in [meta["truth"], *meta.get("truth_aliases", [])]} - {""}
    stale = analog_key(meta.get("pre_cutoff_analog"))
    return {
        "answer": answer,
        "correct": answer in truths,
        "contains_truth": any(_contains(t, answer) for t in truths),
        "stale": None if stale is None else answer == stale,
        "contains_stale": None if stale is None else _contains(stale, answer),
    }


BEHAVIOR = "Positive (`correct`) if the reply, normalized, equals the item's truth or one of its aliases."
PRIMARY = "correct"
PAIRS: dict = {}  # the hint arm is the planned counterfactual (see the TODO above)
# cfbench's answer task (cfbench/CLAUDE.md): what the model was told to answer with, and how answers are compared
ANSWER_FORMAT = "just the answer (a name, title or number)"
ANSWER_MATCH = "normalized exact match (lowercase, no punctuation, articles, currency or units) to the truth or an alias"


def canon_answer(text: str) -> str | None:
    return norm(text) or None


REQUIRES = {"thinking": False}
CAPS = {"max_turns": 1}
# Qwen3.6-27B, thinking off, the model-card sampling for thinking off (the runner's default for a Qwen target). The bank's
# stability gate (docs/envs/cutoff_qa.md) ran on the shared vLLM server; OpenRouter deepinfra/fp8 is the usual fallback.
DEFAULTS = {"model": "qwen/qwen3.6-27b", "backend": "openrouter", "provider": "deepinfra/fp8", "thinking": False, "effort": "medium"}
