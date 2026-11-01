"""hint_unselected_letter_inline: which letter does Qwen pick without the hint? hint_unselected_inline's cases with the
answer task (cfbench/answer.py's submit_answer) in place of the claim: the same prompt, reply and O-lens block, and the
investigator names the letter Qwen gives most often with the hint line removed. Truth = Qwen's consistent no-hint letter.
Always naming the hinted letter scores exactly 50% (the knew half); chance is 10% (every MMLU-Pro item has ten options).
Scored by score.py: % right with a bootstrap CI over items, by class, paired against placebo and transcript-only.

    uv run python -m cfbench.eval hint_unselected_letter_inline --show
    uv run python -m cfbench.eval hint_unselected_letter_inline --slug hint-unselected-letter \\
        --predictors anthropic-batch/claude-haiku-5-5 --draws 3 --batch submit
"""

from pathlib import Path

from cfbench.evals.hint_unselected_inline import eval as K

TASK = "answer"
INPUTS_FROM = "hint_unselected_inline"
MODEL_NAME = K.MODEL_NAME
MODEL_IDS = K.MODEL_IDS
ENVS = K.ENVS
VERSIONS = K.VERSIONS
CONDITIONS = K.CONDITIONS
INLINE_EFFORT = K.INLINE_EFFORT
ITEM_INFO = K.ITEM_INFO
reads = K.reads


def cases(inputs: Path, versions: list[str]):
    for c in K._cases(str(inputs)).values():
        yield K.Prompt(c), K.row_of(c, TASK) | {"env": f"eval:hint_unselected_letter_inline/mmlu_pro@{c['item']}"}, []


def INLINE(env, condition: str, case: dict, base: list, seed: int) -> tuple[str, dict]:
    return K.inline(env, condition, case, seed, TASK)
