"""Which letter does Qwen pick without the hint? Single-turn, with the O-lens block at the option newlines in context.

Kylie, 2026-10-28/29: hint_qwen_dsv4_letter's answer task on hint_qwen_dsv4_inline's 976 cases (same inputs run), one
turn, no tools, Gemini 3.8 Flash at effort low: `transcript_inline`, `olens_inline`, `olens_placebo_inline`
(cfbench/evals/hint_qwen_dsv4/inline.py). Everything but the conditions comes from hint_qwen_dsv4_letter.

    uv run python -m cfbench.eval hint_qwen_dsv4_letter_inline --show --limit 1
"""

from cfbench.evals.hint_qwen_dsv4_letter import eval as L
from cfbench.evals.hint_qwen_dsv4_letter.eval import *

CONDITIONS = ("transcript_inline", "olens_inline", "olens_placebo_inline")
INPUTS_FROM = "hint_qwen_dsv4"
DEFAULT_VERSIONS = L.DEFAULT_VERSIONS
