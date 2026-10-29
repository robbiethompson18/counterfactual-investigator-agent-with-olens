"""knowledge_cutoff: short factual questions about events after Qwen3.6-27B's cutoff (about January 2025), thinking off.

WorkspaceBench's family of the same name (camilablank/workspace-bench, `evals/knowledge_cutoff/`; bank on HF at
`banks/wsb/knowledge_cutoff.json`, built by wsb/build_banks.py): 109 items whose label is Qwen's own stable wrong answer,
usually the stale 2024 value. Here the same questions are an env, so Qwen's samples (the stability gate), any other model's
answers (the investigator census: does it share Qwen's staleness?) and the O-lens reads are ordinary env runs, and the
investigator task is the cfbench eval `knowledge_cutoff` (predict Qwen's reply from the prompt alone). Decisions and
results: docs/envs/knowledge_cutoff.md.

One arm, `recall` (their p1 template: "{question} If you are not sure, give your best guess. Answer with just the
<answer_type>."). No MCQ: shown the bank's 5 options Qwen switches to the true one on about a third of items (the 2026-10-29
smoke), so a multiple-choice render changes the subject's answer. Measures: `answer` (the one bank option the reply names, by
the bank's alias regex), `stale` (= the bank's gold, Qwen's answer: the primary), `correct` (= the truth, when known).
"""

from functools import cache

from environments._lib.core import Ctx, Episode
from wsb import banks
from wsb.knowledge_cutoff import OptionForms

ARMS = ["recall"]


@cache
def _bank() -> dict[str, dict]:
    return {it["id"]: it for it in banks.load("knowledge_cutoff")["items"]}


def items():
    return list(_bank())


def build(arm, item, ctx: Ctx | None = None) -> Episode:
    it = _bank()[str(item)]
    meta = {k: it[k] for k in ("gold", "options", "option_units", "label_type", "answer_type", "stratum")}
    meta |= {"truth": (it.get("truth") or {}).get("canonical")}
    return Episode(messages=[{"role": "user", "content": it["user"]}], meta=meta)


def named_option(meta: dict, text: str) -> str | None:
    """The one bank option the reply names; None if none or several."""
    named = OptionForms.build(meta["option_units"]).named(text)
    return next(iter(named)) if len(named) == 1 else None


def measure(ep) -> dict:
    last = ep["messages"][-1]
    text = (last.get("content") or "") if last["role"] == "assistant" else ""
    meta = ep.get("meta") or {}
    answer = named_option(meta, text)
    return {
        "answer": answer,
        "stale": None if answer is None else answer == meta["gold"],
        "correct": None if answer is None or not meta.get("truth") else answer == meta["truth"],
    }


PRIMARY = "stale"
PAIRS: dict = {}  # no counterfactual: the investigator predicts the reply itself (cfbench/evals/knowledge_cutoff)
REQUIRES = {"thinking": False}
CAPS = {"max_turns": 1}
# WorkspaceBench's gate sampled Qwen at T 1, top-p 1 (the bank header); the OpenRouter copy is fp8, our vLLM copy bf16
DEFAULTS = {
    "model": "qwen/qwen3.6-27b",
    "backend": "openrouter",
    "provider": "deepinfra/fp8",
    "thinking": False,
    "effort": "medium",
    "body": "openrouter",
    "sampling": {"off": {"temperature": 1.0, "top_p": 1.0}},
}
BEHAVIOR = "Positive if the reply names the bank's gold: Qwen3.6-27B's own stable answer to the question (usually a stale value)."
ANSWER_FORMAT, ANSWER_MATCH = "just the answer (a name or number)", "the bank's alias regex: the one option the answer names"
