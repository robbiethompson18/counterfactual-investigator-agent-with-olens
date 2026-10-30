# mcq_unselected: what will Qwen say? An unselected MMLU-Pro set, and the 2k/2k split

**Status (2026-10-29):** ACTIVE. Full runs done on 2,523 cases with Haiku 5.5: the O-lens and the
J-lens each add a small, replicated gain over prompt-only; a linear probe on the same activations
reads far more; the NLA adds the least (results below). Cases rebuilt the same day with the strict
bare-letter parse: 1,251; the first build's 1,285 are superseded (see Census and cases).

## Why

The hint evals ([envs/hint_cf.md](envs/hint_cf.md), `cfbench/evals/hint_qwen_dsv4*`) select
questions three times over: Qwen3.6-27B consistent, DSv4 Flash consistent and disagreeing with Qwen,
and Qwen following the hint. The disagreement filter makes the questions harder than average (Qwen
56% right on them vs 68% on the bank) and imports DSv4's letter bias as a class signal: DSv4's
consistent errors sit on questions whose gold is A or B (122 and 103 of them on the 2,200 bank vs 20
at J), so an early hinted letter alone predicts "Qwen knew" with AUROC 0.70 on the filtered set, and
0.53 with no filter. Robbie, 2026-10-29: keep only the consistency filter, drop the hint, and ask
the plain question: given the prompt, what letter will Qwen say?

## The split (earmarked data)

Drawn once by `cfbench/evals/mcq_unselected/eval.py` (`SEED` 0) and written to the inputs run
`runs/cfbench_evals/2026-10-29_mcq-unselected-inputs/split.json` on HF:

- `random.Random(0).sample(items(), 4000)` over all 12,032 MMLU-Pro test questions in
  `environments/hint_mcq/env.py`'s item order (all 14 categories, not only the seven of the 2,200
  bank).
- **Eval half: 2,000 questions**, the first 2,000 of the sample that no probe has trained on.
- **Probe half: 2,000 questions**, the rest, which includes the 236 sampled questions a probe had
  already trained on. Train probes here, not on the eval half.
- Probe-trained questions, 718 in all: the MCQ answer probe's balanced set
  ([mcq-answer-probe.md](mcq-answer-probe.md), ids from `runs/mcq_answer_probe/2026-10-28_olens-nl`)
  and the refit probe's 496 train items (`runs/mcq_answer_probe/2026-10-28_eval-baseline-12k`, a
  subset of the 718).
- Overlap with earlier work is fine and expected: the sample is over the same 12,032 questions the
  hint evals drew from. Only probe training is kept off the eval half.

## Batch 2 (2026-10-29 evening)

Robbie: "draw fresh". `SEED2` = 1 draws 2,000 more questions from the 8,032 in neither half (none
probe-trained), version `mmlu_pro_b2`: 1,272 consistent cases under the strict parse (1,034 right),
no overlap with batch 1 or the probe half. Reads:
`runs/olens_reads/2026-10-29_mcq-unselected-nohint-b2` (three sharded pods, six layers, over a
1,294-prompt request file written before the strict rebuild; every strict case is in it). The inputs
run with both batches is `runs/cfbench_evals/2026-10-29_mcq-unselected-inputs-b2`
(`--versions mmlu_pro` / `mmlu_pro_b2` select a batch); pool them by case, same prompt, same
harness.

## Census and cases

Qwen3.6-27B, thinking off, no hint, 8 samples per question, from the four runs that together cover
all 12,032 questions: `envs/hint_mcq/2026-10-26_or_nothink2`, `2026-10-27_scale2k`,
`2026-10-27_scale2k-topup`, `2026-10-28_all12k-nohint`, read from their labels file
`labels/measure@4f098526d544-strict.jsonl` (`LABELS` in eval.py, named, not the newest file). A
reply counts only when it is one bare letter (`hint_mcq`'s `measure()` since 2026-10-29); a worked
solution is unscored and counts against consistency. A question is a case when at least 7/8 of its
samples gave the same bare letter. No other selection.

|                                         | count       |
| --------------------------------------- | ----------- |
| eval half                               | 2,000       |
| with census answers                     | 2,000       |
| consistent (>= 7/8), i.e. cases         | 1,251       |
| of which Qwen's modal answer is correct | 1,031 (82%) |
| of which wrong                          | 220         |

So "guess the gold" scores 82% here; the interesting cases are the 220. Report overall accuracy as
the headline and the wrong-subset accuracy as the diagnostic, without selecting on it.

**The first build (1,285 cases: 1,044 right, 241 wrong) is superseded.** It read Qwen's replies with
`hint_mcq`'s old parser, the first standalone capital A-J anywhere in the reply (CODE_SMELL P1 #1).
4,785 of the 96,462 census replies (5%) are worked solutions instead of a letter, and in those the
first capital is often the article "A", the pronoun "I" or a variable. Rebuilt from the same census
with the strict parse:

|                 | first build | strict rebuild |
| --------------- | ----------- | -------------- |
| cases           | 1,285       | 1,251          |
| Qwen right      | 1,044       | 1,031          |
| Qwen wrong      | 241         | 220            |
| removed / added |             | 34 / 0         |
| letter changed  |             | 0              |

- The 34 removed (21 wrong, 13 right): 23 had no bare-letter reply at all (8 worked solutions, which
  the old parser called consistent on their first capital; 14 of the 34 old letters were A, 4 were
  I); 11 had some bare letters, fewer than 7 the same.
- None added and no letter changed, by construction: a kept case has at least 7 bare letters, which
  both parsers read the same. Every case has n = 8; 11 have one worked solution beside 7 letters.
- Over all 12,032 questions the same rule gives 7,943 consistent questions (1,709 wrong), down from
  8,129 (1,821 wrong).
- Why the share is over all samples, not the scored replies only: scored-only gives 1,306 eval-half
  cases, 54 of them on one to six bare letters (12 on a single one), and changes 3 letters. Against
  a reference that reads each worked solution's last letter, that set has 32 bad cases (wrong
  letter, or not 7/8 consistent), more than the old parser's 23; this rule has none.
- The own O-lens reads (`runs/olens_reads/2026-10-29_mcq-unselected-nohint`, requested for the first
  build) cover all 1,251 cases.

## The task

Single turn, no tools but `submit_answer` (`core.INLINE_CONDITIONS`). The eval's default is Gemini
3.8 Flash at effort low (`INLINE_EFFORT`); every full run below used Claude Haiku 5.5 at effort
medium (`--predictors anthropic/claude-haiku-5.5 --effort medium`, PR #383), matching
cutoff_qa_inline.

- `transcript_inline`: the exact no-hint prompt, the setup (8 samples, agreed >= 7 times), "predict
  the letter the model gives, which may or may not be the correct one". The subject is never named
  (`MODEL_NAME`; the runner asserts the id never reaches the predictor). No transcript exists; the
  condition name is cfbench's.
- `olens_inline`: plus O-lens reads at the newline ending each option and at the end of the prompt
  (the instruction's ".", `<|im_end|>`, `\n`, `<|im_start|>`, `assistant`, `\n`), layers
  32/40/48/52/56/60 (cutoff_qa_inline's six), every bullet with its FVE, from
  `runs/olens_reads/2026-10-29_mcq-unselected-nohint` (space lens, T 1; 32-52 in the first job,
  56/60 topped up the same day).
- `olens_placebo_inline`: the same block with another question's own reads at the same positions
  (same option count, seeded, never the question's own), from the same reads run. The two placebo
  smokes below predate the reads and drew from Kylie's hinted-prompt pools instead.

- `jlens_inline` / `jlens_placebo_inline` (PR pending): the J++ lens's top 3 next tokens with their
  probabilities at the same 16 positions and six layers, from
  `runs/jpp_reads/2026-10-29_mcq-unselected-nohint` (read at every token; shown at the O-lens
  block's positions because every token was a ~30k-token prompt). Placebo as for the O-lens.
- `nla_inline` / `nla_placebo_inline`: the all-layer NLA's explanation at the same positions and
  layers with its critic FVE, from `runs/nla_reads/2026-10-29_mcq-unselected-nohint`.

Truth is Qwen's modal letter; scored as % right (`cfbench.answer`). Non-LLM bars: the linear probe
(`runs/mcq_answer_probe/2026-10-29_mcq-unselected-probe`, PR #421: one-hot listwise, `nl`
hidden_states[52], diag standardisation, lambda 1, trained on the 1,360 consistent probe-half
questions, never an eval question) and Qwen's own letter-logit argmax as the ceiling.

## Results (2026-10-29, both batches pooled, Haiku 5.5 medium, 3 draws)

2,523 cases: 2,065 where Qwen's letter is the gold, 458 where it isn't. Accuracy = the submitted
letter equals Qwen's modal letter, over draws. Runs `cfbench/2026-10-29_mcq-unselected-haiku-b1r`,
`-haiku-b2`, `-haiku-jlens-b1`, `-haiku-jlens-b2`, `-haiku-nla-b1`, `-haiku-nla-b2` (NLA at one
draw: its block is ~18k tokens a call, so three draws would have been ~$31). $45 of Haiku in all.

| reader                             | all      | Qwen right | Qwen wrong |
| ---------------------------------- | -------- | ---------- | ---------- |
| Haiku, prompt only                 | 82.6     | 94.4       | 29.5       |
| Haiku + O-lens                     | 84.3     | 95.6       | 33.3       |
| Haiku + O-lens placebo             | 81.7     | 93.4       | 29.0       |
| Haiku + J-lens                     | 84.9     | 95.1       | 38.9       |
| Haiku + J-lens placebo             | 72.3     | 82.5       | 26.6       |
| Haiku + NLA (1 draw)               | 83.4     | 95.0       | 31.4       |
| Haiku + NLA placebo (1 draw)       | 82.2     | 93.9       | 29.3       |
| linear probe, hidden_states[52]    | **91.9** | 94.1       | **82.1**   |
| Qwen's own letter logits (ceiling) | 99.8     | 99.9       | 99.3       |

![prompt only vs O-lens, J-lens, NLA and their placebos, the linear probe and Qwen's logits; all / Qwen right / Qwen wrong](img/mcq-unselected.png)

![paired differences between conditions with 95% CIs, all cases and Qwen-wrong cases](img/mcq-unselected-diff.png)

(`uv run python -m graphs.mcq_unselected` makes both; 95% bootstrap CIs over cases.)

Paired by case, 95% bootstrap over cases (3,000 resamples):

| comparison                   | all                 | Qwen wrong         |
| ---------------------------- | ------------------- | ------------------ |
| O-lens − prompt only         | +1.6 [+0.9, +2.4]   | +3.7 [+1.7, +5.8]  |
| O-lens − O-lens placebo      | +2.6 [+1.9, +3.3]   | +4.3 [+2.3, +6.5]  |
| O-lens placebo − prompt only | −1.0 [−1.5, −0.4]   | −0.6 [−2.1, +0.9]  |
| J-lens − prompt only         | +2.3 [+1.5, +3.0]   | +9.4 [+6.8, +12.0] |
| J-lens − O-lens              | +0.6 [−0.1, +1.4]   | +5.7 [+3.1, +8.3]  |
| J-lens placebo − prompt only | −10.3 [−11.2, −9.4] | −3.0 [−5.0, −0.9]  |
| NLA − prompt only            | +0.8 [−0.0, +1.6]   | +1.9 [−0.6, +4.3]  |
| NLA − NLA placebo            | +1.3 [+0.5, +2.1]   | +2.2 [+0.0, +4.6]  |
| J-lens − NLA                 | +1.5 [+0.6, +2.4]   | +7.5 [+4.7, +10.3] |

The two batches agree (batch 1: O-lens − prompt +2.1 [+1.1, +3.1]; batch 2, the fresh draw: +1.2
[+0.2, +2.2]; on Qwen-wrong +3.8 and +3.6).

What it says:

- **A small, real, replicated gain from the O-lens**, concentrated where it must be: on the 458
  cases where Qwen is wrong, where the investigator's own knowledge points at the gold. The placebo
  block costs a point rather than helping, unlike on the knowledge-cutoff eval.
- **The J-lens reads the letter better** (38.9% vs 33.3% on Qwen-wrong): its top tokens after an
  option line are often literally the letter (`" B" 0.39` at layer 48), where the O-lens writes a
  sentence that may or may not name one. But its placebo is a disaster (−10pp): another question's
  letter tokens are followed as if they were this question's. "J-lens minus its placebo" is not a
  usable number; compare the J-lens with prompt-only.
- **The NLA is the weakest of the three** (+0.8pp over prompt-only, CI touching zero; +1.3 over its
  placebo). Its explanations at the option newlines are long, confident and often about a different
  question (a moon-gravity item gets paragraphs on time dilation and Doppler shift, FVE up to 0.45);
  only the end-of-prompt readouts tend to carry the answer. One draw, so the CIs are wider.
- **Most of the signal is left on the table.** A linear probe on the same positions and layer reads
  Qwen's letter on 82% of the Qwen-wrong cases; the best lens gets the investigator to 39%, the
  O-lens to 33%, from a prompt-only 30%. The metamodels recover about a sixth (J-lens) and a
  fourteenth (O-lens) of what a probe reads from the same activations.
- **Doctrine, not arithmetic.** By category (pooled, O-lens − prompt only): math −0.1, physics 0.0,
  chemistry −0.5; history +5.9, health +4.0, law +3.4, engineering +2.6, economics +2.5. The lens
  carries which doctrine Qwen is applying, not which number it computed.
- **Clean wins are rare.** Only 2 of batch 1's 220 Qwen-wrong cases are 3/3 for the O-lens and 0/3
  for both prompt-only and placebo; the gain is mostly partial nudges. One of the two, a business
  arithmetic question (Qwen B, gold H), Haiku's rationale: "the arithmetic gives $50, which is not
  among the options, so the model must pick a distractor. The end-of-prompt reads in the middle and
  upper layers (48 to 60) repeatedly show 'Answer: B'": O-lens draw
  [Docent](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/bb60e266-ed5d-4f0c-9fa4-74f13f548592),
  prompt-only draw
  [Docent](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/6d3b8b33-bb5a-4de9-b0ce-66954928b7b2).
  The other, computer science (Qwen C, gold A):
  [O-lens](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/c5f13070-8a7f-483e-97b5-af26dc66fda2),
  [prompt-only](https://docent.transluce.org/dashboard/b5ef5858-2956-4ef9-9e97-6edae2ec023e/agent_run/f08147f4-d49d-4763-8f2e-6e0a7e9ae159).
- On Qwen-wrong cases the investigator still submits the gold in over half its draws in every
  condition (prompt-only 56%, O-lens 53%); the lens moves a few percent of draws from gold to Qwen's
  letter.

## Runs

- `runs/cfbench_evals/2026-10-29_mcq-unselected-inputs-strict`: split + cases from the strict-parse
  census, current ("strict" is the parse; the `params.strict` flag is cfbench's, unused here). The
  split is the first build's, unchanged. `latest_inputs` picks the newest inputs run by local mtime,
  so pass `--inputs` on a machine that downloaded both.
- `runs/cfbench_evals/2026-10-29_mcq-unselected-inputs`: SUPERSEDED, the first build (old parser);
  identical to `2026-10-29_mcq-say-inputs`, made before the eval was renamed from `mcq_say`.
- `runs/cfbench/2026-10-29_mcq-say-placebo-smoke10`: 10 cases of the first build, placebo condition
  only, 1 draw, for Robbie to review the prompt in Docent before anything else runs. 2 of its 10
  (10008, 10067) are not cases in the strict rebuild.

- Investigator runs (Haiku 5.5 medium, 3 draws): `runs/cfbench/2026-10-29_mcq-unselected-haiku-b1r`
  (batch 1; `-haiku-b1` is the same slug's first attempt, abandoned on a `--versions` bug before any
  prediction), `-haiku-b2`, `-haiku-jlens-b1`, `-haiku-jlens-b2`, `-haiku-nla-b1`, `-haiku-nla-b2`.
  All on Docent (collection `cfbench`).
- Reads: `runs/olens_reads/2026-10-29_mcq-unselected-nohint` (batch 1, 117,966 rows: 32-52 then a
  56/60 top-up), `-nohint-b2` (batch 2, 119,118 rows, three sharded pods, one replaced for a
  thermally throttled host), `runs/jpp_reads/2026-10-29_mcq-unselected-nohint` (every token, six
  layers), `runs/nla_reads/2026-10-29_mcq-unselected-nohint` (all-layer NLA, critic-scored). About
  $75 of H200 time for everything, of which ~$4 was a pod left idle after its reader finished.
- Probe: `runs/mcq_answer_probe/2026-10-29_mcq-unselected-probe` (`predictions.jsonl` in cfbench's
  answer-task format, conditions `probe` and `qwen_logits`; `cfbench.answer` scores it; the env
  viewer doesn't list it because it isn't under `runs/cfbench/`).

## Open items

- NLA has one draw; two more would cost ~$20 and only narrow a ~1pp effect.
- Mirror the probe run under `runs/cfbench/` so the env viewer shows it beside the Haiku bars.
- Tooling notes from the read campaign: `runpod/olens_reads.sh fetch` truncates the local files if
  the pod is already gone (write to a temp name first); `runpod/watch_pod.sh`'s pre-delete fetch
  uses rsync, which the vLLM image lacks; a reads file on the NC-1 volume once began with 5 MB of
  NUL bytes (an NFS hole) and had to be stripped.
