# Why the O-lens writes Chinese

**Status (2026-10-26):** REFERENCE. Explains an artefact of the old lens, fixed by the space lens.

> **Fixed (2026-10-22):** Agam retrained the lens with the leading space kept,
> `olens_s3d_space_rl600` (Chinese in 3% of bullets vs 36%, same WorkspaceBench score). Use it; this
> doc is about the old `olens_s3d_rl600`. On our 3 favourite transcripts the space lens has Chinese
> in 0.3% of bullets vs 51% (`data/runs/olens_space_fix/`).

The agu18dec O-lens (`agu18dec/olens_and_ar`, `olens_s3d_rl600`) sprinkles Chinese into its readouts
of English transcripts. In our banks (`data/banks/`, measured 2026-10-21 with the Qwen3.6-27B
tokenizer):

| bank        | Chinese tokens | English tokens | ideographs / (ideographs + Latin letters) | bullets with an ideograph |
| ----------- | -------------- | -------------- | ----------------------------------------- | ------------------------- |
| `agam/`     | 4.5%           | 69%            | 2.1%                                      | 34%                       |
| `olens_rl/` | 5.1%           | 65%            | 2.5%                                      | 34%                       |

It is mostly code-switching inside English bullets; fully Chinese bullets are rare. Measure on
decoded text: counting each token that decodes to U+FFFD as Chinese also counts the box-drawing
characters and emoji the lens emits, and overstates it (an earlier count said 7.1% for `olens_rl`).

## Evidence that it is a training artifact

We reproduced it with Robbie's own O-lens on Qwen3.6-27B
([oracle-lens-reconstruction](https://github.com/robbiethompson18/oracle-lens-reconstruction),
writeups `docs/strip-ablation.md` and `docs/zh-probe.md`; outputs on HF
[`…-pilot-strip-ablation`](https://huggingface.co/robbiethompson2018/oracle-lens-qwen3.6-27b-pilot-strip-ablation)
and
[`…-pilot-zh-probe`](https://huggingface.co/robbiethompson2018/oracle-lens-qwen3.6-27b-pilot-zh-probe)).
The suspect is a detail of the agu18dec recipe: each bullet reaches the AR as the text after `- `
with **no leading space** (`olens/README.md`). Our reconstructor was trained on phrases that keep
their leading space, so stripping it pushes English phrases off-distribution. CJK phrases have no
leading space and are unaffected.

20 RL steps from the same SFT checkpoint, scored greedily on held-out English (ultrachat) inputs
(256 rows for the ultrachat legs, 128 for the WildChat legs):

| RL activations                      | strip leading space | phrases with an ideograph | FVE   |
| ----------------------------------- | ------------------- | ------------------------- | ----- |
| ultrachat (English only)            | no                  | 0%                        | 0.160 |
| ultrachat (English only)            | yes                 | 0%                        | 0.134 |
| WildChat (18% of rows from Chinese) | no                  | 1.4%                      | 0.180 |
| WildChat (18% of rows from Chinese) | yes                 | **74%**                   | 0.136 |

- Neither ingredient alone does it. Stripping needs some Chinese in the RL data to seed Chinese
  phrases, and then rewards them on English rows too. agu18dec's RL used WildChat-1M activations and
  fed bullets to the AR without the leading space, i.e. both ingredients.
- The Chinese is on topic: for a "thanks for your work on the project" email the strip lens writes
  `项目紧张时期，我会调整工作模式`. So it carries content, just in the wrong language.
- On Chinese inputs our lenses mostly translate into English; RL raises Chinese there from 0.4%
  (SFT) to 10–14% of phrases.
- Caveats: one seed, 20 steps (agu18dec ran 600), our reconstructor rather than agu18dec's AR.
  Untested: whether agu18dec's AR scores English phrases lower without the leading space (cheap
  check, no training), and whether his inverter stage contributes.

## What this means here

Treat Chinese in a readout as a language artifact, not as evidence that Qwen is "thinking in
Chinese". Translate it and read it as content. Don't score a hypothesis up or down because a bullet
is in Chinese.
