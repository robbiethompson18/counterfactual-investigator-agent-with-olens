# Hint flip probe: would removing this element change Qwen's letter?

**Status (2026-10-30):** NULL. Stopped after the labels: removing a frame line almost never flips a
consistent answer (0.47%), and hint removal repeats the hint_qwen_dsv4 knew/didn't evals and the
hint-first probe ([mcq-answer-probe.md](mcq-answer-probe.md)). The arms and pipeline stay.

Kylie's question (2026-10-29): for each element of a hinted multiple-choice prompt (the hint, a
frame line on top), "if we removed it, would Qwen3.6-27B's letter change?", predicted by a linear
probe from the activations of the prompt WITH the element only (zero counterfactuals). Labels come
from Qwen's bf16 letter argmax on both prompts.

## Design

- **Prompts, per question (15).** No hint under three frames: none, the chat-app JSON line on top
  (`jsonapp`), an eval header (`MMLU-Pro Q<id>` / `TruthfulQA Q<id>`). Per hint source (the Stanford
  professor in each env's own wording; "Wikipedia says it's X."), always naming the correct letter:
  after the options, before the question, under the jsonapp line after / before, as the jsonapp
  JSON's `note`, under the header after. Arms: hint_mcq `nohint`, `jsonapp/nohint`, `mmlu/nohint`,
  `{,before/,jsonapp/,jsonapp-before/,jsonnote/,mmlu/}{hint,wiki}+d1`; hinted_truthfulqa `control`,
  `jsonapp/control`, `header/control`,
  `{hint,hint_before,jsonapp/hint,jsonapp/hint_before,jsonnote,header/hint}/{stanford,wikipedia}` on
  `tqa-N.h<gold>` items. Old arms rebuild byte-identical; the TruthfulQA `measure()` gives identical
  labels on 41k stored episodes.
- **Removal cases, per question (20).** The hint from each of the 12 hinted prompts (to the no-hint
  prompt of the same frame; the JSON note to `jsonapp` no hint); the frame from jsonapp+after,
  jsonapp+before, header+after (× 2 sources) and from the two framed no-hint prompts (to plain).
- **Questions, two pools.** `balanced`: MMLU-Pro, 500 Qwen gets wrong + 500 right (strict no-hint
  census modal letter) from the probe half of the [mcq_unselected](mcq-unselected.md) split, never
  the eval halves; TruthfulQA, all 144 questions Qwen's control-n8 modal answer gets wrong + 144
  right. `twomodel`: the hint_qwen_dsv4 pool (`twomodel_qwen_dsv4_follow.csv`) minus 64 already in
  the balanced draw and 36 MMLU-Pro items in an eval half: 236 MMLU-Pro + 9 TruthfulQA. 1,533
  questions, 22,995 prompts, 30,660 cases.
- **Forward pass.** One HF bf16 pass per prompt (`scripts/hs_at_positions.py`, the slow torch
  kernels: with flash-linear-attention 4% of option-newline sites at `hidden_states[52]` diverged,
  cosine down to 0.47, on a 40-prompt check), residuals at `hidden_states` 32–60 at the option
  newlines, element ends and last 10 tokens, and letter log-probs at the answer position. ~7
  prompts/s on one H200; ~66 min of RunPod H200 in all (~$6).
- **Consistency** (Kylie, 2026-10-30, as the other hint evals' ≥ 7/8 samples): a prompt is
  consistent if its letter logits, tempered to the envs' sampling (T 0.7, top-p 0.8), give the
  argmax letter p with P(Binomial(8, p) ≥ 7) ≥ 0.5; a case counts when both prompts are. Against the
  8-sample census on 1,236 MMLU-Pro no-hint prompts this call agrees 91.8% of the time (839 both,
  296 neither, 31 logits only, 70 census only).

## Results

Full run `2026-10-29_hint-flip` (`labels.md`, `frames.txt`). Flip rates (bf16 argmax; consistent =
both prompts consistent):

| removed                     | all cases                 | consistent cases       |
| --------------------------- | ------------------------- | ---------------------- |
| hint                        | 6,780 / 18,396 (36.9%)    | 2,383 / 11,828 (20.1%) |
| frame                       | 781 / 12,264 (6.4%)       | **46 / 9,883 (0.47%)** |
| frame: header, no hint      | 13.4%                     | 1.83%                  |
| frame: jsonapp, no hint     | 11.5%                     | 0.81%                  |
| frame: header, hint after   | 5.6% / 1.2% (prof / wiki) | 0.08% / 0.07%          |
| frame: jsonapp, hint after  | 8.3% / 2.2%               | 0.52% / 0.21%          |
| frame: jsonapp, hint before | 4.0% / 4.6%               | 0.23% / 0.46%          |

- **Frame flips are wavering answers.** 735 of the 781 frame flips (94%) have Qwen inconsistent on
  at least one of the two prompts; unfiltered, frame flips are 12.0% on questions Qwen gets wrong
  and 1.2% on ones it gets right. By pool (consistent): balanced 0.46%, twomodel 0.48%; by dataset:
  MMLU-Pro 0.40%, TruthfulQA 0.71%. A frame line moves Qwen only between letters it was already
  split over, so on consistent answers there is nothing for a probe to predict.
- **Hint removal flips (20% of consistent cases), but that is the knew/didn't question** the
  hint_qwen_dsv4 evals and the hint-first probe tune already answer.
- **Sanity vs the fp8 sampled framing deltas** (README of hint_mcq, wrong questions, gold hint): the
  smoke's hint-follow ordering matched (before ≥ after > header > JSON note > jsonapp+after), argmax
  a little above T 0.7 sampling.
- **Smoke probes (84 questions, unfiltered, no conclusions):** hint removal, a final-token logistic
  probe reached AUROC 0.88 at `hidden_states` 48–52 on the 50:50 set, tied with a logistic on Qwen's
  own letter-pmf features + arm one-hots (0.89; Δ −0.01 [−0.06, +0.04]); frame removal, Qwen's
  letter margin alone 0.95 vs the probe 0.86. Not rerun on the full set.

## Where things live

- Code: [`environments/hint_mcq/flip.py`](../environments/hint_mcq/flip.py) (`prep`, `labels`,
  `split` for a held-out investigator-eval question set, `probe` with baselines, the consistency
  filter, stacked arm/hand variants and transfer; built, never run on the full set),
  [`scripts/hs_at_positions.py`](../scripts/hs_at_positions.py) (`--layers`, `--chunk`, `--no-fla`),
  [`scripts/sky/hs_positions.yaml`](../scripts/sky/hs_positions.yaml).
- HF runs: `runs/hint_flip_probe/2026-10-29_hint-flip` (questions, prompts, cases, labels, frames)
  and `…-smoke` (+ the smoke `probe.md`). Activations (28 GB) only in `data/probe_cache/<run>/` on
  Kylie's bramble.

```bash
uv run --with transformers --with jinja2 python -m environments.hint_mcq.flip prep [--smoke 25]
python scripts/hs_at_positions.py RUN/hs_inputs.jsonl data/probe_cache/<run> --layers 32,40,48,52,56,60 --no-fla --chunk 2000
uv run python -m environments.hint_mcq.flip labels RUN
```
