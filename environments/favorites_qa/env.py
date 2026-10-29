"""favorites_qa: free-response preference questions ("What is your favorite constellation? Answer in one word.") asked of
Qwen3.6-27B as a bare user turn, no system prompt, thinking off: WorkspaceBench's basic_readout `implicit` items scaled up
(Robbie, 2026-10-29). There is no truth; the label for the predict-Qwen task is Qwen's own gated modal answer.

Bank v1: questions written by Sonnet agents in Claude Code from scripts/favorites/BRIEF.md (one category per question, asked
once, in one of seven shapes: favorite, least_favorite, pick_one, recommend, be_one, name_one, choose_for), merged and gated by
scripts/favorites/bank.py (the gate is scripts/cutoff_gate.py: 8 samples at T 1 + greedy on the shared server). Each item has
`qwen_modal`, `qwen_modal_count`, `qwen_greedy`, `gate_class` (stable, unstable, refusal_or_null) and `gate_run`. items() is
the stable (>= 7/8) items; `FAVORITES_QA_ALL=1` takes every item. Banks are data: HF `banks/favorites_qa/<version>.json`,
fetched to data/banks/favorites_qa/ on first use.

Arms: `nohint` only: the question verbatim. DEFAULTS are cutoff_qa's (shared server, T 1.0 / top_p 1.0, thinking off), so
env transcripts are sampled like the gate's. Measures (no judge): `answer` = the reply normalized with cutoff_qa's `norm`;
`modal` (PRIMARY) = it equals `qwen_modal` normalized the same way; `contains_modal` = the modal answer is a whole phrase in it.
"""

import json
import os
from functools import cache

from counterfactual_investigator_agent_with_olens.runs import REPO
from environments._lib.core import Ctx, Episode
from environments._lib.runfolder import REPO_ROOT
from environments.cutoff_qa import env as cutoff_qa

ARMS = ["nohint"]
VERSION = os.environ.get("FAVORITES_QA_BANK", "v3")  # v1: 290-question pilot; v2: pilot + wave 1 (one gate); v3: v2 + wave 2
ALL_ITEMS = os.environ.get("FAVORITES_QA_ALL") == "1"
norm = cutoff_qa.norm
DEFAULTS, REQUIRES = cutoff_qa.DEFAULTS, cutoff_qa.REQUIRES  # the shared server, T 1 / top_p 1, thinking off
META_KEYS = ("domain", "category", "shape", "qwen_modal", "qwen_modal_count", "qwen_greedy", "gate_class", "gate_run")


@cache
def bank(version: str = VERSION) -> dict:
    path = REPO_ROOT / f"data/banks/favorites_qa/{version}.json"
    if not path.exists():
        from huggingface_hub import hf_hub_download

        hf_hub_download(REPO, f"banks/favorites_qa/{version}.json", repo_type="dataset", local_dir=REPO_ROOT / "data")
    return json.loads(path.read_text())


@cache
def _items() -> dict[str, dict]:
    return {it["id"]: it for it in bank()["items"]}


def items():
    return [i for i, it in _items().items() if ALL_ITEMS or it["gate_class"] == "stable"]


def build(arm, item, ctx: Ctx | None = None) -> Episode:
    it = _items()[item]
    return Episode(messages=[{"role": "user", "content": it["question"]}], meta={k: it[k] for k in META_KEYS} | {"bank": VERSION})


def measure(ep) -> dict:
    reply = next((m.get("content") or "" for m in reversed(ep["messages"]) if m["role"] == "assistant"), "")
    answer, modal = norm(reply), norm(ep["meta"]["qwen_modal"])
    return {"answer": answer, "modal": answer == modal, "contains_modal": cutoff_qa._contains(modal, answer)}


BEHAVIOR = "Positive (`modal`) if the reply, normalized, equals Qwen's gated modal answer to the question."
PRIMARY = "modal"
PAIRS: dict = {}
ANSWER_FORMAT = "just the answer, as the question asks (one word or a short name)"
ANSWER_MATCH = "normalized exact match (lowercase, no punctuation or leading article) to Qwen's modal answer"


def canon_answer(text: str) -> str | None:
    return norm(text) or None


CAPS = {"max_turns": 1}
