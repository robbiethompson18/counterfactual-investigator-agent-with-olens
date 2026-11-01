# chive_inline: on CHIVE-style claims the model's reply adds about 1 point of AUROC, and no lens beats its placebo

**Status (2026-10-31):** REFERENCE. One run per set, Haiku 5.5 only; CHIVE's exact prompt (with its
Question line) was not run, so Haiku vs Opus mixes the predictor with that prompt change.

CHIVE's yes/no benchmark (the official eval sets in
[adamkarvonen/chive-data](https://huggingface.co/datasets/adamkarvonen/chive-data), all three
targets, 4,816 claims, balanced true/false): each claim says an intervention moves a behaviour's
rate by at least 30pp. CHIVE's transcript-only arm reads the conversation, the model's reply, a "why
did you ..." Question line, the behaviour definition, the baseline rate and the intervention in
prose. They never ran an arm without the reply. We ran Haiku 5.5 (effort low, 3 draws, single turn)
on their stored prompt minus the Question line (`transcript_inline`), and on the same minus the
model's reply too (`no_transcript_inline`, edit only). Code: `cfbench/evals/chive_inline`. Run:
`runs/cfbench/2026-10-31_chive-haiku` on HF (28,895 of 28,896 predictions valid).

| target       | claims | Haiku edit only | Haiku transcript | transcript − edit only  | CHIVE Opus transcript | CHIVE Opus resampling |
| ------------ | -----: | --------------: | ---------------: | ----------------------- | --------------------: | --------------------: |
| Qwen3-8B     |  1,870 |           0.756 |            0.759 | +0.003 [−0.006, +0.012] |                 0.794 |                 0.949 |
| Gemma 3 27B  |  1,637 |           0.704 |            0.723 | +0.020 [+0.008, +0.031] |                 0.775 |                 0.935 |
| Qwen3.5-397B |  1,309 |           0.756 |            0.772 | +0.016 [+0.003, +0.029] |                 0.815 |                 0.959 |
| Pooled       |  4,816 |           0.737 |            0.749 | +0.012 [+0.006, +0.019] |                 0.792 |                 0.947 |

AUROC over every prediction (Pooled: one AUROC over all three targets' claims); 95% bootstrap CIs
over claims, the difference paired by claim.

Most of CHIVE's transcript-only score comes from the edit's wording, the behaviour line and the
stated baseline, not from reading the model's reply. Chart: `uv run python -m graphs.chive_inline`
(`docs/img/chive-inline.png`, `chive-inline-diff.png`).

## With lenses: Agam's Qwen3.6-27B incidents

CHIVE's targets have no O-lens, NLA or J++, so the lens test runs on Agam's 300 Qwen3.6-27B
incidents ([agu18dec/chive-forensics](https://huggingface.co/datasets/agu18dec/chive-forensics),
thinking off), the half at or under the median read-position count (150 incidents, 515 claims,
CHIVE's 50pp / 15pp labels: 316 true, 199 false). Same prompt as above, except the intervention is a
unified diff of the conversation (Agam's per-experiment summaries state the result). Reads are of
the original conversation and reply only, every punctuation and end-of-turn token plus the ten
tokens before the reply, layers 32 to 60, all bullets shown; a placebo is another incident's reads
laid on this one's positions. Haiku 5.5, effort medium, 3 draws. Code:
`cfbench/evals/chive_agam_inline`. Runs:
`runs/cfbench/2026-10-31_chive-agam-haiku-{nolens,olens2,nla5,jlens}`; reads
`runs/{olens,nla,jpp}_reads/2026-10-31_chive-agam-median`.

| condition  | AUROC | − transcript                           | − its placebo           |
| ---------- | ----: | -------------------------------------- | ----------------------- |
| edit only  | 0.775 |                                        |                         |
| transcript | 0.780 | +0.005 over edit only [−0.014, +0.022] |                         |
| O-lens     | 0.788 | +0.008 [−0.008, +0.025]                | +0.001 [−0.015, +0.015] |
| NLA        | 0.784 | +0.005 [−0.012, +0.022]                | +0.004 [−0.010, +0.020] |
| J++        | 0.791 | +0.011 [−0.003, +0.025]                | +0.003 [−0.010, +0.015] |

Placebo AUROCs: O-lens 0.787, NLA 0.780, J++ 0.788. Every lens adds about a point over
transcript-only, and so does its placebo from an unrelated incident: the gain is from extra text in
the prompt, not from what the lens read. Chart:
`uv run python -m graphs.chive_agam_inline --runs <the four runs>`
(`docs/img/chive-agam-inline.png`, `chive-agam-inline-diff.png`).
