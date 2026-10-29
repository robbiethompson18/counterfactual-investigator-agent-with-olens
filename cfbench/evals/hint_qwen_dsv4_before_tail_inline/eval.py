"""Did Qwen know it? hint_qwen_dsv4_before_inline (hint first) plus O-lens reads at the end of the prompt.

Kylie, 2026-10-29 (the main one): hint_qwen_dsv4_before_inline's 952 cases with hint_qwen_dsv4_tail_inline's second block
(the "." ending the prompt, <|im_end|>, "\\n", <|im_start|>, "assistant", "\\n") and the fine-details sentence.

    uv run python -m cfbench.eval hint_qwen_dsv4_before_tail_inline --show --limit 1 \\
        --inputs data/runs/cfbench_evals/2026-10-29_hint-qwen-dsv4-inputs-before
"""

from cfbench.evals.hint_qwen_dsv4_before_inline.eval import *
from cfbench.evals.hint_qwen_dsv4_tail_inline.eval import INLINE  # noqa: F401  (the tail block)
