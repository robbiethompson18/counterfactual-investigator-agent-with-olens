"""Which letter does Qwen pick without the hint? hint_qwen_dsv4's cases with the answer task instead of the claim.

Kylie, 2026-10-28.

The same balanced, pooled (MMLU-Pro + TruthfulQA) case set and inputs run as hint_qwen_dsv4
(INPUTS_FROM): questions Qwen3.6-27B and DSv4 Flash answer
consistently and differently, half Qwen got right without the hint and half it got wrong, balanced within each gold
letter; the base arm hints the correct letter and the investigator sees K.N_SHOWN runs that picked it. Here the investigator names
the letter Qwen gives with no hint (cfbench/answer.py's submit_answer, any option); truth = Qwen's modal no-hint letter.
Always naming the hinted letter scores exactly 50% (the "knew" half); chance is about 1 / number of options. Scored by
score.py: % right with a bootstrap CI over items, by class, and the paired condition comparisons.

    uv run python -m cfbench.eval hint_qwen_dsv4_letter --show --versions mmlu_pro
    uv run python -m cfbench.eval hint_qwen_dsv4_letter --slug hint-qwen-dsv4-letter --draws 3
    uv run python -m cfbench.evals.hint_qwen_dsv4_letter.score data/runs/cfbench/<date>_<slug>
"""

import json
import re
from collections import Counter
from pathlib import Path

from cfbench.core import Rollout
from cfbench.evals.hint_qwen_dsv4 import eval as K
from environments._lib.core import load_env

ENVS = K.ENVS
VERSIONS = K.VERSIONS
MODEL_NAME = None
CONDITIONS = K.CONDITIONS
OLENS = K.OLENS
TASK = "answer"
INPUTS_FROM = "hint_qwen_dsv4"
QWEN = "Qwen/Qwen3.6-27B"
reads = K.reads
ITEM_INFO = K.ITEM_INFO
ARM_LABELS = K.ARM_LABELS
PAIR_LABELS = K.PAIR_LABELS
LINKS = K.LINKS


LETTERS = "ABCDEFGHIJKLMNOP"
OPTION = re.compile(r"^\(?([A-P])[.)] (.*)$", re.MULTILINE)  # "A. text" (hint_mcq) or "(A) text" (hinted_truthfulqa)
# in order: "the answer is B" / "option B" / "answer: (B)", "(B)", "**B**", a leading "B" ("B", "B.", "B) ...", "B - ..."),
# a lone capital letter anywhere except the word "I"
PATTERNS = [
    re.compile(r"(?i:\b(?:answer|option|choice|letter)\b(?:\s+(?:is|would be|should be|will be))?)\s*[:=-]?\s*\(?([A-P])\b(?![a-z'])"),
    re.compile(r"\(([A-P])\)"),
    re.compile(r"\*\*\(?([A-P])\)?\*\*"),
    re.compile(r"^\W*([A-P])(?:[.):,]|\s*$|\s+[-–—])"),
    re.compile(r"(?<![A-Za-z'])([A-HJ-P])(?![A-Za-z'])"),
]
ANSWER_MATCH = (
    'the chosen letter: "the answer is X" / "option X", "(X)", "**X**", a leading X, else a lone capital letter (not the word '
    '"I"); with no letter, the one option whose text the answer contains'
)


def canon(text: str, opts: list[str]) -> str | None:
    """The letter an answer names (PATTERNS in order, only letters of this question), else the one option whose text it
    contains (longest wins; a tie is no answer)."""
    t = (text or "").strip()
    for pat in PATTERNS:
        for m in pat.finditer(t):
            if LETTERS.index(m.group(1)) < len(opts):
                return m.group(1)
    low = t.lower().rstrip(".")
    hits = sorted(((len(o), LETTERS[k]) for k, o in enumerate(opts) if o and o.lower().rstrip(".") in low), reverse=True)
    if hits and (len(hits) == 1 or hits[0][0] > hits[1][0]):
        return hits[0][1]
    return None


class Prompt(K.Prompt):
    """hint_qwen_dsv4's prompt env plus the answer-task attributes of the item's env (cfbench/core.py)."""

    def __init__(self, version: str, item: str, gold: str):
        super().__init__(version, item, gold)
        mod = load_env(VERSIONS[version]["env"])
        self.ANSWER_FORMAT, self.ANSWER_MATCH = mod.ANSWER_FORMAT, ANSWER_MATCH
        opts = [o.strip() for _, o in OPTION.findall(mod.build(VERSIONS[version]["arms"][1], item).messages[-1]["content"])]
        self.canon_answer = lambda text: canon(text, opts)


def cases(inputs: Path, versions: list[str]):
    """(prompt env, case row fields, base runs) per case: truth = Qwen's modal no-hint letter."""
    for line in (inputs / "cases.jsonl").read_text().splitlines():
        c = json.loads(line)
        if c["version"] not in versions:
            continue
        V, env = VERSIONS[c["version"]], Prompt(c["version"], c["item"], c["gold"])
        base = [Rollout(source=s, messages=m, label=True, answer=env.canon_answer(m[-1]["content"] or "")) for s, m in c["shown"]]
        modal, k = Counter(c["answers_cf"]).most_common(1)[0]
        row = {"version": "pooled", "source": c["version"], "item": c["item"], "class": c["class"], "gold": c["gold"],
               "model": QWEN, "env_name": V["env"],
               "env": f"eval:hint_qwen_dsv4_letter/{c['version']}@{c['item']}", "task": TASK, "base_arm": V["arms"][0],
               "cf_arm": V["arms"][1], "answers_base": dict(Counter(r.answer for r in base)), "answers_cf": c["answers_cf"],
               "modal_cf": modal, "k_cf": k, "n_cf": sum(c["answers_cf"].values())}  # fmt: skip
        yield env, row, base
