"""Did Qwen know it? hint_qwen_dsv4_inline with the hint before the question.

Kylie, 2026-10-29: hint_qwen_dsv4_inline's 976 cases (same items, classes and no-hint truth) and single-turn conditions,
but the base arm puts the professor sentence first (hint_mcq `before/hint+d1`, hinted_truthfulqa `hint_before/stanford`;
VERSIONS before_*), so the option newlines the O-lens reads come after the hint and no longer equal the no-hint prompt's.
Qwen3.6-27B bf16 x8 per item (hint_mcq / hinted_truthfulqa run 2026-10-29_hint-before-pool), one following run shown
per case; reads in K.READS_NL_BEFORE.

    uv run python -m cfbench.eval hint_qwen_dsv4 --prep --versions before_mmlu_pro,before_truthfulqa,before_mmlu_pro_12k --tag before
    uv run python -m cfbench.eval hint_qwen_dsv4_before_inline --show --limit 1
"""

from cfbench.evals.hint_qwen_dsv4 import eval as K
from cfbench.evals.hint_qwen_dsv4_inline.eval import *

CONDITIONS = ("transcript_inline", "olens_inline", "olens_placebo_inline")
INPUTS_FROM = "hint_qwen_dsv4"
DEFAULT_VERSIONS = K.BEFORE_VERSIONS
