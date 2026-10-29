"""supergpqa: SuperGPQA questions, thinking off, one arm, no hint (Robbie, 2026-10-29).

The question for this env is not a counterfactual: it is whether Qwen3.6-27B's no-CoT answer on a harder MCQ pool is
consistent and often wrong, so that "guess the answer Qwen gives without seeing it" has more usable items than MMLU-Pro
(hint_mcq's nohint arm: 66% correct, 32 of 200 items consistently wrong). SuperGPQA (Du et al., 2025, arXiv 2502.14739) is
26,529 mostly 10-option questions over 285 disciplines; Qwen3.6-27B scores 66% on it with thinking on (its model card).

Items: 10,000 of the 15,300 non-calculation questions (`is_calculation` false; on calculation items Qwen writes out the
working instead of a letter, which is CoT in the reply): items 0:1000 are the original seeded sample (the 2026-10-29_n8
run), 1000:10000 a second seeded sample from the rest (the scale-up). The prompt is hint_mcq's nohint prompt: the
question, lettered options, "Answer with the letter only." Measures: `answer` (the letter, from a reply that is just a
letter; a reply with any more text is unscorable, `verbose`), `correct` (= gold, the primary).
"""

import json
import re
import shutil
from functools import cache

import pandas as pd

from environments._lib.core import Ctx, Episode
from environments._lib.runfolder import REPO_ROOT

ARMS = ["base"]
INSTRUCTION = "Answer with the letter only."
LETTERS = "ABCDEFGHIJ"
JSONL = REPO_ROOT / "data/evalcue_cache/supergpqa_all.jsonl"  # m-a-p/SuperGPQA, fetched on first use
N_ITEMS = 1000  # the original seeded sample (items 0:1000, the 2026-10-29_n8 run)
N_EXTRA = 9000  # the scale-up: a second seeded sample from the rest, items 1000:10000


@cache
def _table() -> pd.DataFrame:
    if not JSONL.exists():
        from huggingface_hub import hf_hub_download

        JSONL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(hf_hub_download("m-a-p/SuperGPQA", "SuperGPQA-all.jsonl", repo_type="dataset"), JSONL)
    pool = pd.DataFrame([json.loads(line) for line in JSONL.open()])
    pool = pool[~pool.is_calculation]
    base = pool.sample(n=N_ITEMS, random_state=0)  # unchanged: the same ids as the stored run
    extra = pool[~pool.uuid.isin(base.uuid)].sample(n=N_EXTRA, random_state=1)
    df = pd.concat([base, extra], ignore_index=True)
    return df.set_index("uuid", drop=False)


def items():
    return list(_table().uuid)


def build(arm, item, ctx: Ctx | None = None) -> Episode:
    row = _table().loc[item]
    meta = {
        "gold": row.answer_letter,
        "discipline": row.discipline,
        "field": row.field,
        "difficulty": row.difficulty,
        "n_options": len(row.options),
    }
    question = row.question.strip() + "\n" + "\n".join(f"{c}. {o}" for c, o in zip(LETTERS[: len(row.options)], row.options, strict=True))
    return Episode(messages=[{"role": "user", "content": question + "\n\n" + INSTRUCTION}], meta=meta)


LETTER = re.compile(r"^\(?([A-J])[).]?\s*$")  # the whole reply is one letter


def first_letter(text: str) -> str | None:
    m = LETTER.match(text.strip())
    return m.group(1) if m else None


def measure(ep) -> dict:
    reply = ep["messages"][-1].get("content") or ""
    answer = first_letter(reply)
    if answer is None:
        return {"correct": None, "verbose": len(reply.strip()) > 3}
    return {"answer": answer, "correct": answer == ep["meta"]["gold"], "verbose": False}


BEHAVIOR = "Positive if the reply's first letter is the gold answer."
PRIMARY = "correct"
PAIRS = {}
ANSWER_FORMAT, ANSWER_MATCH = "the letter only", "the first letter A-J, case-insensitive"


def canon_answer(text: str) -> str | None:
    return first_letter(text.strip().upper())


REQUIRES = {"thinking": False}
CAPS = {"max_turns": 1}
DEFAULTS = {  # hint_mcq's: Qwen3.6-27B on OpenRouter pinned to DeepInfra fp8, thinking off, evalcue's thinking-off sampling
    "model": "qwen/qwen3.6-27b",
    "backend": "openrouter",
    "provider": "deepinfra/fp8",
    "thinking": False,
    "effort": "medium",
    "body": "openrouter",
    "sampling_all_models": True,
    "sampling": {
        "on": {"temperature": 1.0, "top_p": 0.95, "presence_penalty": 0.0},
        "off": {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5},
    },
}
