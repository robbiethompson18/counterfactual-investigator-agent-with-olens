# Qwen-native CHIVE pipeline (sketch, 2026-10-19)

Draft design for regenerating CHIVE-style incidents on Qwen3.6-27B so that every assistant token is
Qwen's and the behaviors are worth explaining. Nothing here is built yet.

## Why

Agam's set ([agu18dec/chive-forensics](https://huggingface.co/datasets/agu18dec/chive-forensics),
held-out 204) has three problems for us:

- **Off-policy history.** 84/204 incidents have earlier assistant turns written by WildChat's
  ChatGPT, not Qwen. Only 57% of assistant turns (65% of assistant characters) are Qwen's.
- **It splits our result.** Transcript-only passes 77% on the 120 pure-Qwen incidents and 60% on the
  84 with ChatGPT turns. O-lens vs transcript is −1.5 pp [−6.2, +3.0] on the first and +9.5 pp
  [+2.4, +16.9] on the second. Post hoc, so a hypothesis, but our only O-lens signal sits on the
  off-policy half.
- **Boring behaviors.** By a rough hand label, ~52% are Qwen being plainly wrong (factual slips,
  self-contradiction), ~36% are coherent but unremarkable content choices, ~12% refusal/scope.

Also: thinking off (Qwen3.6's non-default mode) and a 500-token cap that truncates replies.

## Starting point

CHIVE's pipeline is public: [adamkarvonen/chive](https://github.com/adamkarvonen/chive), data and
per-stage records at
[adamkarvonen/chive-data](https://huggingface.co/datasets/adamkarvonen/chive-data). Stages (paper
appendix B, [arXiv 2608.16747](https://arxiv.org/abs/2608.16747), 2026):

1. **Sample** 30 responses per prompt, temperature 1.0.
2. **Screen**: Opus rates each prompt's responses for unexpectedness (1–5, ≥3 kept, behavior rate
   ≥30%) and freezes a yes/no classifier question.
3. **Investigate**: Opus agent runs ~10–15 counterfactual edits, 30 resamples each, graded by the
   frozen question (Sonnet grader), then files a report.
4. **Verify**: judge scores the report 1–10; keep ≥8.
5. **Claims + filters**: blind writer turns experiments into claims; concreteness filter.

Their cost: ~$3.3k Opus per 4,714-prompt eval run, ~$1.9 per verified investigation. The paper says
the screener prompt and its 185 few-shot examples took the most manual iteration, and that more
capable targets yield fewer concrete behaviors.

Plan: fork it, change the parts below, keep its record schemas so Agam's harness
(`vendor/detective-joracle`) reads the output unchanged.

## What we change

### 1. On-policy prompts only

Every assistant turn in a prompt must be sampled from Qwen3.6-27B. Sources, in order of priority:

- **PETRI against Qwen3.6-27B.** An auditor model simulates the user and tools; the target writes
  every assistant turn. Truncate at each assistant turn, as CHIVE did. Multi-turn, on-policy,
  alignment-flavored scenarios (pressure, deception opportunities, agentic tools). CHIVE ran 173
  seeds × 2 and got ~1.7–2k prompts per model.
- **WildChat first user turn only.** Drop all later turns. Real user distribution, trivially
  on-policy. Same dedup and length filters as CHIVE.
- **WildChat multi-turn, re-rolled.** Keep user turn 1, sample Qwen's reply, have an LLM write the
  next user turn conditioned on Qwen's reply (seeded by the original user's later turns). The user
  side becomes synthetic; the assistant side is on-policy. Only if the first two don't give enough
  multi-turn incidents.

### 2. Sampling settings

- **Thinking: open question.** On is Qwen3.6's default, and it is where the O-lens should matter
  most: the CoT is in the transcript, so the interesting cases are causes the CoT never states.
  Costs: more tokens per sample and per O-lens bank, and CoT makes the transcript baseline stronger.
  Could run both and keep them as separate splits.
- **No truncated replies.** Raise the cap (e.g. 1,500 response tokens, 1,024 thinking) and drop
  samples that still hit it, instead of grading cut-off text.

### 3. Retarget the screener

The screener decides what we end up caring about, so this is the main design work.

- Down-weight plain capability errors (wrong fact, arithmetic slip, mis-parsed code) unless a
  specific prompt feature triggers them.
- Up-weight behavior classes with alignment interest: sycophancy and caving, misreading or
  overriding user intent, persona and role drift, refusals with odd triggers, self-presentation
  (claims about itself), eval awareness, unprompted moralizing, deception or concealment in PETRI.
- Rebuild the few-shot pool from incidents we would actually want to read. Start from the held-out
  204 labels in the "Content choice" and "Policy / scope" tabs of the hard-cases artifact, plus
  hand-picked PETRI hits.
- **Don't filter on transcript-predictability.** "Keep only incidents a transcript-only agent gets
  wrong" is the collider bias from `CLAUDE.md`. Select on behavior class and cause concreteness,
  never on how hard the incident is for one of our arms.

### 4. Investigation and verification

Unchanged from CHIVE: Opus investigator, Sonnet grader, verifier ≥8, blind claim writer. Resamples
go to our own vLLM server (RunPod), at the same settings as stage 1.

### 5. O-lens banks

Agam's banks came from his viewer. For new incidents we generate them ourselves with the O-lens
checkpoint ([agu18dec/olens_and_ar](https://huggingface.co/agu18dec/olens_and_ar)), same layout as
`data/banks/olens_rl/` (layers 20–60 step 4, k=4). Needs a GPU job per incident; size TBD.

### 6. Splits up front

Fix dev / val / test by prompt before running any investigator arm on them. Report PETRI and
WildChat separately.

## Pilot

Small enough to judge yield and quality in about a day:

1. Serve Qwen3.6-27B with vLLM on RunPod.
2. 300 WildChat first-turn prompts + PETRI on ~30 seeds.
3. Sample, screen with the retargeted screener, read every flagged behavior by hand.
4. Investigate the best ~30, then run transcript-only vs O-lens with Agam's harness.

Go/no-go: at least ~1 in 10 prompts yields a behavior we would want to read about, and the
investigations verify at a rate near CHIVE's.

## Rough costs

Order-of-magnitude guesses, to be replaced with pilot measurements:

- **Opus:** ~$2 per verified investigation, so ~$600 for 300 kept incidents, plus screening.
- **GPU:** resamples dominate. ~12 experiments × 30 samples × ~1k tokens ≈ 360k generated tokens per
  investigation, ~100M for 300 investigations. That's hours to a day on one or two H100s, depending
  on vLLM throughput for a 27B dense model.
- **Time:** about a week to the first usable split, most of it screener iteration.

## Risks

- **Yield.** Qwen3.6-27B may produce few interesting behaviors with concrete causes; CHIVE saw yield
  drop with model capability. The pilot measures this.
- **PETRI scenarios are synthetic.** On-policy for the assistant, but the user side is an auditor
  model, so the setting is less "in the wild".
- **O-lens training distribution.** If the lens was trained on non-thinking text, readouts over
  thinking tokens may be worse. Check before committing to thinking on.

## Alternative worth knowing about

CHIVE already released on-policy-ish Qwen3-8B data, including a thinking run and a PETRI run
(`qwen3_8b_wildchat_thinking_eval`, `qwen3_8b_petri_eval` in chive-data). If we had an O-lens for
Qwen3-8B (a first step in the Final Pitch), PETRI incidents there could be a cheap test set without
running the pipeline at all. Still the WildChat screener's behavior mix, though.
