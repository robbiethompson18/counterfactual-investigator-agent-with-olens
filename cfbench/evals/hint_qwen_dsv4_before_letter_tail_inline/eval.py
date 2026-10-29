"""Which letter does Qwen pick without the hint? hint_qwen_dsv4_before_letter_inline (hint first) plus O-lens reads at the
end of the prompt (hint_qwen_dsv4_tail_inline's block and fine-details sentence).

    uv run python -m cfbench.eval hint_qwen_dsv4_before_letter_tail_inline --show --limit 1 \\
        --inputs data/runs/cfbench_evals/2026-10-29_hint-qwen-dsv4-inputs-before
"""

from cfbench.evals.hint_qwen_dsv4_before_letter_inline.eval import *
from cfbench.evals.hint_qwen_dsv4_letter_tail_inline.eval import INLINE  # noqa: F401  (the tail block)
