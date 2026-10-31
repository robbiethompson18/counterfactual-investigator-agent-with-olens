"""cutoff_qa: short-answer factual questions whose answer changed around Qwen3.6-27B's knowledge cutoff (~January 2025),
each asked as a bare user turn with no system prompt and thinking off, the way WorkspaceBench's knowledge_cutoff family
renders its items (Robbie, 2026-10-29).

Bank v4 (default; `CUTOFF_QA_BANK=v3` / `v2` / `v1` for the older banks): 6,411 present-tense, dateless questions ("Who is
the CEO of Intel?"): v2's 803 (629 rewrites of v1's dated questions plus 174 fame-steered ones), 4,664 written by 100 Sonnet
agents from disjoint entity lists (brief v4) and 944 from 100 more agents given changed-facts-first lists (brief v5), every one
gated on the shared Qwen server (8 samples at T 1 + greedy): `qwen_modal` (Qwen's modal normalized answer), `qwen_modal_count`,
`gate_class` (stable_right, stable_wrong, unstable, refusal_or_null). By default items() is the 2,605 stable (>= 7/8) items
(861 stable-wrong); `CUTOFF_QA_ALL=1` takes every item. The label for the predict-Qwen task is `qwen_modal`; `truth` (the
writer's answer at authoring time), `truth_now` (a web recheck as of 2026-10-29, stable v2 items only; for entity-list items
equal to `truth`) and `pre_cutoff_analog` are metadata. docs/envs/cutoff_qa.md has the provenance, the gate numbers and the
weaknesses; `scripts/cutoff_facts/merge.py` built v1 from the subagents' files; the gate is scripts/cutoff_gate.py (PR #382).
Banks are data: HF `banks/cutoff_qa/<version>.json`, fetched to data/banks/cutoff_qa/ on first use.

Arms: `nohint` only (hints are ruled out for this env): the user message is the bank's `question` string verbatim (it ends
"If you are not sure, give your best guess. Answer with just the <answer_type>."). DEFAULTS target the shared Qwen server
(bf16) at temperature 1.0, top_p 1.0, no presence penalty, thinking off: WorkspaceBench's knowledge_cutoff sampling and the
stability gate's (runs/cutoff_gate/2026-10-29_*), so O-lens reads of env transcripts match the gate's. PRIMARY is `correct`;
`stale` is the field downstream code uses for the predict-Qwen task.

Measures (no LLM judge anywhere): `answer` = the reply normalized (`norm`: lowercase, accents and punctuation stripped,
leading article dropped, currency / units / digit-group commas stripped from numbers, whitespace collapsed); `correct` = it
equals the normalized truth or an alias (PRIMARY); `stale` = it equals Qwen's gated modal answer `qwen_modal` (v1 items, which
have none: the pre-cutoff analog; None without either); `contains_truth` / `contains_stale` = the same as a whole phrase inside
the reply, for replies that add words.
"""

import json
import os
import re
import unicodedata
from collections import Counter
from functools import cache

from counterfactual_investigator_agent_with_olens.qwen import MODEL as SERVED_NAME
from counterfactual_investigator_agent_with_olens.runs import REPO
from environments._lib.core import Ctx, Episode
from environments._lib.runfolder import REPO_ROOT

ARMS = ["nohint"]
VERSION = os.environ.get("CUTOFF_QA_BANK", "v4")  # v1: 1,072 dated; v2: 803 present-tense, gated; v3: v2 + 4,664 entity-list; v4: + 944
ALL_ITEMS = os.environ.get("CUTOFF_QA_ALL") == "1"  # include unstable and refusal items too
META_KEYS = ("truth", "truth_aliases", "pre_cutoff_analog", "event_date", "domain", "answer_type", "numeric")
GATE_KEYS = ("qwen_modal", "qwen_modal_count", "qwen_greedy", "gate_class", "gate_run", "truth_now", "truth_changed", "source_batch")
GATE_KEYS += ("category",)  # entity-list items (v3, v4): the list's category


@cache
def bank(version: str = VERSION) -> dict:
    path = REPO_ROOT / f"data/banks/cutoff_qa/{version}.json"  # HF banks/cutoff_qa/<version>.json, fetched on first use
    if not path.exists():
        from huggingface_hub import hf_hub_download

        hf_hub_download(REPO, f"banks/cutoff_qa/{version}.json", repo_type="dataset", local_dir=REPO_ROOT / "data")
    return json.loads(path.read_text())


@cache
def _items() -> dict[str, dict]:
    return {it["id"]: it for it in bank()["items"]}


def items():
    """The stable (>= 7/8 samples agree) items of the bank, or every item with CUTOFF_QA_ALL=1 (v1 has no gate: every item)."""
    return [i for i, it in _items().items() if ALL_ITEMS or it.get("gate_class", "stable").startswith("stable")]


def build(arm, item, ctx: Ctx | None = None) -> Episode:
    it = _items()[item]
    meta = {k: it[k] for k in META_KEYS} | {k: it[k] for k in GATE_KEYS if k in it} | {"bank": VERSION}
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


@cache
def _gate_samples(gate_run: str) -> dict[str, dict]:
    path = REPO_ROOT / "data/runs/cutoff_gate" / gate_run.removeprefix("cutoff_gate/") / "samples.jsonl"
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def gate_label(gate_run: str, item: str, modal: str) -> str:
    """The predict-Qwen label: the gate's modal answer in `norm` form, taken from the raw samples. The gate clusters samples
    with its own normalizer (scripts/cutoff_gate.py: "N'Djamena" -> "n djamena"), and norm() of that string keeps the space
    where norm() of the answer itself drops the apostrophe ("ndjamena"), so a predictor saying exactly what Qwen says was
    marked wrong on 17 of 1,849 tool-set labels (fixed 2026-10-31; favorites_qa stores a raw sample for the same reason)."""
    r = _gate_samples(gate_run)[item]
    forms = Counter(norm(t) for t, n in zip(r["samples"], r["samples_norm"], strict=True) if n == modal)
    return forms.most_common(1)[0][0]


def analog_key(analog: str | None) -> str | None:
    """A v1 analog's comparison form: the text before any parenthetical ("sonoma (macOS 14, 2023)" -> "sonoma")."""
    return norm(analog.split("(")[0]) or None if analog else None


def stale_key(meta: dict) -> str | None:
    """What a stale reply equals: Qwen's gated modal answer, or for v1 items (no gate) the writer's pre-cutoff analog."""
    return norm(meta["qwen_modal"]) or None if meta.get("qwen_modal") else analog_key(meta.get("pre_cutoff_analog"))


def _contains(phrase: str, text: str) -> bool:
    return re.search(rf"(?<!\S){re.escape(phrase)}(?!\S)", text) is not None


def measure(ep) -> dict:
    meta = ep.get("meta") or {}
    reply = next((m.get("content") or "" for m in reversed(ep["messages"]) if m["role"] == "assistant"), "")
    answer = norm(reply)
    truths = {norm(t) for t in [meta["truth"], *meta.get("truth_aliases", [])]} - {""}
    stale = stale_key(meta)
    correct = answer in truths
    return {
        "answer": answer,
        "positive": not correct,  # Qwen is wrong; ambiguous: also counts off-script wrong answers and alias misses, not only outdated facts
        "correct": correct,
        "contains_truth": any(_contains(t, answer) for t in truths),
        "stale": None if stale is None else answer == stale,
        "contains_stale": None if stale is None else _contains(stale, answer),
    }


BEHAVIOR = (
    "Positive (`correct`) if the reply, normalized, equals the item's truth or one of its aliases; "
    "`stale` = it equals Qwen's gated modal answer; `positive` = not correct."
)
PRIMARY = "correct"
PAIRS: dict = {}  # one arm, no counterfactual pair
# cfbench's answer task (cfbench/CLAUDE.md): what the model was told to answer with, and how answers are compared
ANSWER_FORMAT = "just the answer (a name, title or number)"
ANSWER_MATCH = "normalized exact match (lowercase, no punctuation, articles, currency or units) to the truth or an alias"


def canon_answer(text: str) -> str | None:
    return norm(text) or None


REQUIRES = {"thinking": False}
CAPS = {"max_turns": 1}
# Qwen3.6-27B bf16 on the shared vLLM server ($QWEN_VLLM_URL, served as "qwen"; runpod/CLAUDE.md), thinking off, sampled at
# temperature 1.0 / top_p 1.0 with no presence penalty: WorkspaceBench's knowledge_cutoff sampling and the bank's stability
# gate's (runs/cutoff_gate/2026-10-29_* on HF), so later O-lens reads are on transcripts sampled the same way.
DEFAULTS = {
    "model": "Qwen/Qwen3.6-27B",
    "backend": "vllm",
    "url": os.environ.get("QWEN_VLLM_URL"),
    "served_name": SERVED_NAME,
    "thinking": False,
    "effort": "medium",
    "sampling_all_models": True,
    "sampling": {"temperature": 1.0, "top_p": 1.0},
}
