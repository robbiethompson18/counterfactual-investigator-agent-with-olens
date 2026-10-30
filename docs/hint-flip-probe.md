# Hint flip probe: would removing this element change Qwen's letter?

**Status (2026-10-29):** ACTIVE. Code and question selection done; GPU run pending.

Kylie's question: for each element of a hinted multiple-choice prompt (the hint, a frame line on
top), "if we removed it, would Qwen3.6-27B's letter change?", predicted by a linear probe from the
activations of the prompt WITH the element only (zero counterfactuals). Labels come from Qwen's
bf16 letter argmax on both prompts.

## Design

- **Prompts, per question (15).** No hint under three frames: none, the chat-app JSON line on top
  (`jsonapp`), an eval header (`MMLU-Pro Q<id>` / `TruthfulQA Q<id>`). Per hint source (the
  Stanford professor in each env's own wording; "Wikipedia says it's X."), always naming the
  correct letter: after the options, before the question, under the jsonapp line after / before, as
  the jsonapp JSON's `note`, under the header after. Arms: hint_mcq `nohint`, `jsonapp/nohint`,
  `mmlu/nohint`, `{,before/,jsonapp/,jsonapp-before/,jsonnote/,mmlu/}{hint,wiki}+d1`;
  hinted_truthfulqa `control`, `jsonapp/control`, `header/control`,
  `{hint,hint_before,jsonapp/hint,jsonapp/hint_before,jsonnote,header/hint}/{stanford,wikipedia}`
  on `tqa-N.h<gold>` items. Old arms rebuild byte-identical; the TruthfulQA `measure()` gives
  identical labels on 41k stored episodes.
- **Removal cases, per question (20).** The hint from each of the 12 hinted prompts (to the no-hint
  prompt of the same frame; the JSON note to `jsonapp` no hint); the frame from jsonapp+after,
  jsonapp+before, header+after (× 2 sources) and from the two framed no-hint prompts (to plain).
- **Questions: two pools.** `balanced`: MMLU-Pro, 500 Qwen gets wrong + 500 right (strict no-hint
  census modal letter, no consistency or difficulty filter) from the probe half of the
  [mcq_unselected](mcq-unselected.md) split, never the eval halves; TruthfulQA, all 144 questions
  Qwen's control-n8 modal answer gets wrong + 144 right. `twomodel`: the hint_qwen_dsv4 pool
  (`environments/hint_mcq/twomodel_qwen_dsv4_follow.csv`, Qwen and DSv4 Flash consistent and
  disagreeing), minus 64 already in the balanced draw and 36 MMLU-Pro items in an eval half: 236
  MMLU-Pro + 9 TruthfulQA. 1,533 questions, 22,995 prompts, 30,660 removal cases.
- **Forward pass.** One HF bf16 pass per prompt (`scripts/hs_at_positions.py`), thinking-off
  template, residuals at `hidden_states` 32, 40, 48, 52, 56, 60 at: each option's newline (the
  answer probe's `nl`; on TruthfulQA prompts whose last option ends the user turn, its site is
  `<|im_end|>`, the doc's "original" variant, so no prompt is edited), the element ends (hint
  sentence's last token, frame line's last token), and the last 10 tokens (the instruction's ".",
  `<|im_end|>`, assistant header, empty think block). Letter log-probs at the answer position ("A"…
  tokens; "(A"… on TruthfulQA).
- **Labels.** flip = argmax letter with ≠ without. Also TV distance of the letter pmfs and Δp(hinted
  letter).
- **Balance.** Pooled flip vs no-flip 50:50 overall by subsampling the larger class (Kylie,
  2026-10-29: not per stratum, so a stratum that rarely flips shows the probe saying "doesn't
  matter"). Results on that set and on all cases (natural rates).
- **Probes** (5-fold GroupKFold by question; logistic C by inner 3-fold on fold 0's training cases):
  (1) flip probe, logistic per layer and element type on the final token, the final token +
  element-end token, or the mean of the last 10; (2) counterfactual-answer probe, listwise at the
  option newlines (`hidden_states[52]`) fine-tuned from the answer probe with ‖w − w0‖² (PR #369's
  recipe), target = the letter without the element, predicted flip = 1 − p(observed letter); the
  answer probe is refit on the cached MMLU-Pro set minus every MMLU-Pro question here (disjoint for
  both pools); (3) transfer hint ↔ frame, professor ↔ Wikipedia, after ↔ before, MMLU-Pro ↔
  TruthfulQA.
- **Baselines** (same folds): base rate, answer = hinted letter, Qwen's letter margin / entropy,
  p(hinted letter), letter only (hinted letter + answer letter one-hots), the answer probe's argmax
  ≠ Qwen's letter (+ entropy), stratum only, and a logistic on all hand features + stratum one-hots
  (the headline comparison, ΔAUROC with a bootstrap over questions). Easiness guards: AUROC within
  margin quintiles; within-question AUROC; twomodel split by whether DSv4 is right.

## Results

Pending the GPU run.

## Where things live

- Code: [`environments/hint_mcq/flip.py`](../environments/hint_mcq/flip.py) (`prep`, `labels`,
  `probe`), [`scripts/hs_at_positions.py`](../scripts/hs_at_positions.py) (`--layers`, `--chunk`,
  `--no-fla`), [`scripts/sky/hs_positions.yaml`](../scripts/sky/hs_positions.yaml).
- Runs: `runs/hint_flip_probe/` on HF; activations in `data/probe_cache/<run>/` (not on HF).

```bash
uv run --with transformers --with jinja2 python -m environments.hint_mcq.flip prep [--smoke 25]
python scripts/hs_at_positions.py RUN/hs_inputs.jsonl data/probe_cache/<run> --layers 32,40,48,52,56,60 --chunk 2000
uv run python -m environments.hint_mcq.flip labels RUN
uv run --with scikit-learn python -m environments.hint_mcq.flip probe RUN --transfer
```
