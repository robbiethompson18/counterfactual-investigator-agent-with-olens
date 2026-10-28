# Alignment faking in cfbench: the O-lens is null on DeepSeek V4 Flash

**Status (2026-10-27):** NULL. The investigator does not get better at predicting the preview/public
compliance gap from DSV4 Flash O-lens reads, pooled or per request; nothing in the reads separates
the requests that flip from the ones that never do. Kept so nobody retries it with the same lens.

## Setup

- Env: `environments/alignment_faking` (Greenblatt et al. 2024 helpful-only, no CoT), DeepSeek V4
  Flash thinking off. Arms are the `<deployment_stage>` tag, `public` (unmonitored) vs `preview`
  (trained on). Original prompt: 72/500 vs 22/500 comply. v7 (Kylie's rewording): 64/200 vs 21/200.
  GLM 5.3 has no gap on either, so it is not a target here.
- Lens: Camila Blank's DSV4 Flash inverter (`camila-blank/deepseek-v4-flash-olens-inverter`, layer
  30, greedy + 2 samples, no FVE, no AR), read with `scripts/dsv4_olens/` on our RunPod pods
  (`scripts/dsv4_olens/CLAUDE.md`). Reads on HF: `runs/olens_reads/2026-10-27_dsv4-af-v7` (v7, one
  episode per arm and request, 200 transcripts) and `2026-10-27_dsv4-af-orig-public5` (original, all
  5 public samples per request, 500 transcripts). Placebo:
  `2026-10-27_placebo-glm-email-for-dsv4-af-v7` (GLM email reads cut to 3 bullets).
- Predictor: Gemini 3.8 Flash, harness v7, conditions `no_transcript`, `transcript_agentic`,
  `olens_agentic`, `olens_placebo_agentic`. Labels: the env's regex `measure` (accept tag, no
  refusal after it; see `environments/alignment_faking/README.md` for its known reframe false
  positives).

## Pooled cases: public ↔ preview, RMSE of the predicted rate

| prompt   | no_transcript | transcript_agentic | olens_agentic | olens_placebo_agentic | null |
| -------- | ------------- | ------------------ | ------------- | --------------------- | ---- |
| original | 7.4pp         | 13.1pp             | 52.2pp\*      | 20.8pp\*              | 10pp |
| v7       | 11.7pp        | 22.1pp             | 31.8pp        | 28.2pp                | 22pp |

\* Kylie's 2026-10-27 run (`2026-10-27_af-cf-robbie-dsv4`), public → preview only, base shown as
0/100 refusal-selected runs. Rows without a star: `2026-10-27_af-orig-dsv4`,
`2026-10-27_af-v7-dsv4`, `2026-10-27_af-v7-dsv4-olens`, 5 draws, both directions.

Removing the training cue (preview → public) is easy in every condition (1–9pp). Adding it is where
the error lives: with the lens the predictor says 0.74–0.78 on every draw against a truth of 0.32
(v7) or 0.14 (original); placebo does nearly the same. The prompt alone is the best condition,
because Flash knows the Greenblatt setup and the lens text (real or placebo) unlocks the paper's
number. Transcript tools sit at the null: 200 identical `<rejected/>` replies carry no information.

## Per request: which scenarios flip?

One case per AdvBench request, public → preview. p = the investigator's P(comply in preview); label
= any preview compliance for that request (5 samples original, 2 samples v7); AUROC of p against the
label and Spearman ρ between p and the preview rate. The "first-token logit" row is not an
investigator: it is the model's own log P(accept tag) − log P(reject tag) at the first reply token
of the public-arm run(s), from the reads' `logits.jsonl`.

| condition                       | v7 AUROC | v7 ρ | original AUROC | original ρ |
| ------------------------------- | -------- | ---- | -------------- | ---------- |
| no_transcript                   | 0.68     | 0.34 | 0.65           | 0.30       |
| transcript_agentic              | 0.67     | 0.29 | 0.58           | 0.20       |
| olens_agentic (1.7 draws/item)  | 0.71     | 0.41 | not run        |            |
| olens_placebo_agentic (1.7)     | 0.65     | 0.28 | not run        |            |
| first-token logit, no predictor | 0.76     | 0.45 | 0.76           | 0.46       |

v7: 41 of 100 requests flip; runs `2026-10-27_af-v7-dsv4-peritem`, `-fill`, `-d12` (draw 0 for all
four conditions, draws 1–2 of the lens conditions, cut short by Gemini's daily quota). Original:
`2026-10-27_af-orig-dsv4-peritem-nolens2`. Standard error on one AUROC is about 0.06, so the lens vs
placebo gap (0.06) is inside noise, and no investigator condition beats the model's own logprob.
Kylie's earlier per-item runs on the original prompt with Haiku and Sonnet
(`2026-10-26_af-cf-haiku-L30`, `-sonnet-L30`) gave lens 0.56 / 0.63 vs placebo 0.45 / 0.47, the same
shape.

## What the reads say

Regex features over every public-arm refusal (32 flip vs 66 never on the original prompt, 30 vs 56
on v7): the read at the `public` tag token paraphrases the system prompt ("public, not training")
for flippers and never-flippers alike; the read at the first reply token contains both agree and
refuse language in nearly every transcript; "agree and no refuse" is 0 in both groups; the same
click-automation refusal boilerplate recurs across items. The inverter has no FVE or AR to filter
hallucination with, and no feature we tried separates the groups. The viewer runs are
`runs/olens_reads/2026-10-26_dsv4flash_alignment_faking` (public-arm refusals with the preview
outcome as a label) and `runs/alignment_faking/2026-10-27_olens-reworded`.

## Not run (Gemini quota, 2026-10-27 21:30 PDT)

The original-prompt per-item lens conditions, with all 5 public samples read so the few-reads rule
passes, and a third draw of the v7 lens conditions. The reads exist; the command:

```bash
uv run python -m cfbench.run env_runs:alignment_faking/2026-10-26_dsv4flash-nocot \
  --model deepseek/deepseek-v4-flash --pairs public:preview --items all \
  --conditions olens_agentic,olens_placebo_agentic \
  --olens-reads data/runs/olens_reads/2026-10-27_dsv4-af-orig-public5/cfbench/reads.jsonl,data/runs/olens_reads/2026-10-27_placebo-glm-email-for-dsv4-af-v7/reads.jsonl \
  --draws 1 --concurrency 12 --slug af-orig-dsv4-peritem-lens --description "..."
```

Expect lens ≈ placebo ≈ 0.65 and the logit at 0.76; we stopped because that answer would not change
the conclusion.
