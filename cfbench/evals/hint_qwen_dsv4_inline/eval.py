"""Did Qwen know it? Single-turn: hint_qwen_dsv4's question, with the O-lens block at the option newlines in context.

Kylie, 2026-10-28/29: hint_qwen_dsv4's cases and scoring (`knew`, % right at p >= 0.5), but the investigator answers in one
turn with no tools (cfbench/evals/hint_qwen_dsv4/inline.py, Gemini 3.8 Flash at effort low): `transcript_inline` (the
base-arm prompt and reply), `olens_inline` (plus every O-lens bullet at each option's newline, layers 32/40/48/52) and
`olens_placebo_inline` (another question's bullets with the same number of positions). Its own page so it reads like the
agentic eval's: one pool of 976 cases (mmlu_pro 81+81, truthfulqa 5+5 and mmlu_pro_12k 402+402, the rest of MMLU-Pro
under the same twomodel criteria), inputs run cfbench_evals/2026-10-29_hint-qwen-dsv4-inputs-all976. Everything but
the conditions comes from hint_qwen_dsv4.

    uv run python -m cfbench.eval hint_qwen_dsv4_inline --show --limit 1
    uv run python -m cfbench.eval hint_qwen_dsv4_inline --draws 1 --concurrency 128 --slug hint-qwen-dsv4-inline-...
"""

from cfbench.evals.hint_qwen_dsv4 import eval as K
from cfbench.evals.hint_qwen_dsv4.eval import *

CONDITIONS = ("transcript_inline", "olens_inline", "olens_placebo_inline")
INPUTS_FROM = "hint_qwen_dsv4"
DEFAULT_VERSIONS = K.DEFAULT_VERSIONS
