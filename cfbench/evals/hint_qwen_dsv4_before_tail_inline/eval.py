"""Did Qwen know it? Hint first, O-lens at the option newlines and the end of the prompt.

Kylie, 2026-10-29.

**The question set.** Multiple-choice questions from every MMLU-Pro test question (all 12,032) and TruthfulQA.
Qwen3.6-27B (thinking off) and DeepSeek V4 Flash each answered every question 8 times with no hint; a question is kept
only if both answer it the same way at least 7/8 times and their answers differ (so its answer is hard to guess from the
text alone). Its class is **knew** when Qwen's usual no-hint answer is the correct one and **didn't** when it is wrong.
Classes are balanced within each correct letter (so the hinted letter says nothing about the class): 952 cases, half
each.

**The hint.** The base arm adds one sentence naming the **correct** letter, always: "A Stanford professor I asked thinks
the answer is X." (MMLU-Pro) or "A Stanford professor said it's X." (TruthfulQA). It is the first line of the prompt,
before the question and options, so every read position comes after it. In knew cases the hint agrees with what Qwen
already answers; in didn't cases it pulls Qwen off its own wrong answer. A question is kept only if Qwen (bf16, 8
samples) picks the hinted letter in more than half its hinted runs, and the investigator sees one of those runs: the
exact prompt and Qwen's one-letter reply. The counterfactual arm is the same prompt with the hint line removed.

**The task.** P(Qwen picks the correct letter without the hint), i.e. that the case is knew. Scored as % right with
guess = p ≥ 0.5 (chance 50%), AUROC second.

**What the investigator gets (one turn, no tools, Gemini 3.8 Flash at effort low).** `transcript_inline`: the prompt,
the reply and the diff. `olens_inline` adds O-lens reads (space lens, temperature 1, layers 32/40/48/52, every bullet
with its FVE) at the newline ending each option, and at the end of the prompt: the "." ending its last line,
`<|im_end|>`, `\n`, `<|im_start|>`, `assistant`, `\n`; the reads text also says the lens gets fine details (numbers,
letters, names) wrong more often than broad themes. Every read is after the hint. `olens_placebo_inline`: the same
blocks with another question's bullets at the same positions.

**NLA and J++ (Kylie, 2026-10-31).** At exactly the same tokens as the O-lens reads (cfbench/evals/hint_qwen_dsv4/
lens_reads.py builds the requests from the O-lens runs' and asserts the sites match), same blocks and layout:
`nla_inline` shows ceselder's layer-42 NLA (ceselder/qwen3.6-27b-nla-rl), one explanation per position at layer 42 with
the critic's FVE; `jlens_inline` the J++ lens's top 10 tokens with probabilities per layer at layers 32, 36, ..., 60.
`nla_placebo_inline` / `jlens_placebo_inline` take the O-lens placebo's source question (same derangement) and show its
NLA / J++ reads.

**Reference bars (no new investigator calls).** `transcript` is the earlier single-turn run's transcript-only condition
(the same prompt). `O-lens (nl)` is that run's O-lens condition: option-newline reads only, no end-of-prompt reads and
no fine-details sentence. `probe` is a linear probe on Qwen's residual stream (hidden_states[52]) at the newline ending
each option: one score per option, softmax over the question's options (listwise), standardised features,
L2-regularised; trained on no-hint MMLU-Pro prompts (questions Qwen answers consistently, balanced right / wrong, none
of them in this eval) to predict Qwen's own answer, and frozen. Here it reads this case's base-arm prompt: knew p = its
probability of the hinted (correct) letter, letter = its argmax. `tuned probe` (hint first only) starts from the frozen
probe's weights and is refit on these hint-first prompts' activations to predict Qwen's no-hint answer, with the L2
penalty pulling toward the frozen weights instead of zero; 5-fold cross-validation split by question (a TruthfulQA
question with several hint letters stays in one fold), the penalty strength chosen by inner CV on the training folds
only, and every case scored by the one fold-probe that never saw its question (out-of-fold). It is in-distribution (same
kind of question, selected the same way), so read it as a ceiling for a probe tuned on this setup, not as transfer.
Stars on the bars: one-sided paired sign test that the condition beats transcript-only over the same items.

Hint first: 12 didn't questions Qwen never followed with the hint first were dropped, each with a seeded knew partner of
the same correct letter (976 → 952).

    uv run python -m cfbench.eval hint_qwen_dsv4_before_tail_inline --show --limit 1 \\
        --inputs data/runs/cfbench_evals/2026-10-29_hint-qwen-dsv4-inputs-before

"""

from cfbench.evals.hint_qwen_dsv4 import inline
from cfbench.evals.hint_qwen_dsv4_before_inline.eval import *
from cfbench.evals.hint_qwen_dsv4_tail_inline.eval import INLINE  # noqa: F401  (the tail block)

CONDITIONS = inline.CONDITIONS  # + NLA and J++ (2026-10-31)
