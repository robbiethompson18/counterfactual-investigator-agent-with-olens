"""Did Qwen know it? hint_qwen_dsv4_inline plus O-lens reads at the end of the prompt.

Kylie, 2026-10-29: hint_qwen_dsv4_inline's 976 cases, inputs and single-turn conditions, with a second O-lens block at
the "." that ends the prompt's last line, <|im_end|>, "\\n", <|im_start|>, "assistant", "\\n" (after the hint; layers
32/40/48/52, READS_TAIL), and a sentence that the lens gets fine details (numbers, letters, names) wrong more often than
broad themes (inline.DETAIL_NOTE). The placebo takes both blocks from the same other question.

    uv run python -m cfbench.eval hint_qwen_dsv4_tail_inline --show --limit 1
"""

from cfbench.evals.hint_qwen_dsv4 import eval as K
from cfbench.evals.hint_qwen_dsv4 import inline
from cfbench.evals.hint_qwen_dsv4_inline.eval import *

READS_TAIL = K.REPO_ROOT / "data/runs/olens_reads/2026-10-29_hint-qwen-dsv4-tail"
TAIL = {"tail_reads": READS_TAIL, "detail_note": True}


def INLINE(env, condition: str, case: dict, base: list, seed: int) -> tuple[str, dict]:
    return inline.prompt(env, condition, case, base, seed, K.VERSIONS[case["source"]] | TAIL, K.MODEL_NAME, K.ASK(case))
