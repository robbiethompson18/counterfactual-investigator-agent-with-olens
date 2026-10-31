# chive_inline: on CHIVE's own claims, the model's reply adds 0 to 2 points of AUROC

**Status (2026-10-31):** REFERENCE. One run, Haiku 5.5 only; CHIVE's exact prompt (with its Question
line) was not run, so Haiku vs Opus mixes the predictor with that prompt change.

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

AUROC over every prediction; 95% bootstrap CIs over claims, the difference paired by claim.

Most of CHIVE's transcript-only score comes from the edit's wording, the behaviour line and the
stated baseline, not from reading the model's reply. Chart: `uv run python -m graphs.chive_inline`
(`docs/img/chive-inline.png`, `chive-inline-diff.png`).
