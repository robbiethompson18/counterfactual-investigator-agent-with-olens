"""Which letter does Qwen pick without the hint? hint_qwen_dsv4_letter_inline with the hint before the question.

Kylie, 2026-10-29: hint_qwen_dsv4_before_inline's cases and inputs run (hint first, reads after it) with the answer task.

    uv run python -m cfbench.eval hint_qwen_dsv4_before_letter_inline --show --limit 1
"""

from cfbench.evals.hint_qwen_dsv4 import eval as K
from cfbench.evals.hint_qwen_dsv4_letter_inline.eval import *

CONDITIONS = ("transcript_inline", "olens_inline", "olens_placebo_inline")
INPUTS_FROM = "hint_qwen_dsv4"
DEFAULT_VERSIONS = K.BEFORE_VERSIONS
