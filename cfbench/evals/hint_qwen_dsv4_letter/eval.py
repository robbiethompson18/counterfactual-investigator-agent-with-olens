"""Which letter does Qwen pick without the hint? hint_qwen_dsv4's cases with the answer task instead of the claim.

Kylie, 2026-10-28.

The same balanced case set and inputs run as hint_qwen_dsv4 (INPUTS_FROM): questions Qwen3.6-27B and DSv4 Flash answer
consistently and differently, half Qwen got right without the hint and half it got wrong, balanced within each gold
letter; the base arm hints the correct letter and the investigator sees 5 runs that picked it. Here the investigator names
the letter Qwen gives with no hint (cfbench/answer.py's submit_answer, any option); truth = Qwen's modal no-hint letter.
Always naming the hinted letter scores exactly 50% (the "knew" half); chance is about 1 / number of options. Scored by
score.py: % right with a bootstrap CI over items, by class, and the paired condition comparisons.

    uv run python -m cfbench.eval hint_qwen_dsv4_letter --show --versions mmlu_pro
    uv run python -m cfbench.eval hint_qwen_dsv4_letter --slug hint-qwen-dsv4-letter --draws 3
    uv run python -m cfbench.evals.hint_qwen_dsv4_letter.score data/runs/cfbench/<date>_<slug>
"""

import json
from collections import Counter
from pathlib import Path

from cfbench.core import Rollout
from cfbench.evals.hint_qwen_dsv4 import eval as K
from environments._lib.core import load_env

ENVS = K.ENVS
VERSIONS = K.VERSIONS
MODEL_NAME = None
CONDITIONS = K.CONDITIONS
TASK = "answer"
INPUTS_FROM = "hint_qwen_dsv4"
QWEN = "Qwen/Qwen3.6-27B"
reads = K.reads
LINKS = K.LINKS


class Prompt(K.Prompt):
    """hint_qwen_dsv4's prompt env plus the answer-task attributes of the item's env (cfbench/core.py)."""

    def __init__(self, version: str, item: str, gold: str):
        super().__init__(version, item, gold)
        mod = load_env(VERSIONS[version]["env"])
        self.ANSWER_FORMAT, self.ANSWER_MATCH, self.canon_answer = mod.ANSWER_FORMAT, mod.ANSWER_MATCH, mod.canon_answer


def cases(inputs: Path, versions: list[str]):
    """(prompt env, case row fields, base runs) per case: truth = Qwen's modal no-hint letter."""
    for line in (inputs / "cases.jsonl").read_text().splitlines():
        c = json.loads(line)
        if c["version"] not in versions:
            continue
        V, env = VERSIONS[c["version"]], Prompt(c["version"], c["item"], c["gold"])
        base = [Rollout(source=s, messages=m, label=True, answer=env.canon_answer(m[-1]["content"] or "")) for s, m in c["shown"]]
        modal, k = Counter(c["answers_cf"]).most_common(1)[0]
        row = {"version": c["version"], "item": c["item"], "class": c["class"], "gold": c["gold"], "model": QWEN, "env_name": V["env"],
               "env": f"eval:hint_qwen_dsv4_letter/{c['version']}@{c['item']}", "task": TASK, "base_arm": V["arms"][0],
               "cf_arm": V["arms"][1], "answers_base": dict(Counter(r.answer for r in base)), "answers_cf": c["answers_cf"],
               "modal_cf": modal, "k_cf": k, "n_cf": sum(c["answers_cf"].values())}  # fmt: skip
        yield env, row, base
