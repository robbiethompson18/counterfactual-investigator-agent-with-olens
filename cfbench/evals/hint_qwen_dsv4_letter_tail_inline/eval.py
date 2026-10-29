"""Which letter does Qwen pick without the hint? hint_qwen_dsv4_letter_inline plus O-lens reads at the end of the prompt.

Kylie, 2026-10-29: as hint_qwen_dsv4_tail_inline (the end-of-prompt block and the fine-details sentence) on the letter
task.

    uv run python -m cfbench.eval hint_qwen_dsv4_letter_tail_inline --show --limit 1
"""

from cfbench.evals.hint_qwen_dsv4 import inline
from cfbench.evals.hint_qwen_dsv4_letter import eval as L
from cfbench.evals.hint_qwen_dsv4_letter_inline.eval import *
from cfbench.evals.hint_qwen_dsv4_tail_inline.eval import TAIL


def INLINE(env, condition: str, case: dict, base: list, seed: int) -> tuple[str, dict]:
    return inline.prompt(env, condition, case, base, seed, L.VERSIONS[case["source"]] | TAIL, "Qwen3.6-27B", L.INLINE_ASK)
