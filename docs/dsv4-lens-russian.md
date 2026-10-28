# Why the DeepSeek V4 Flash O-lens writes Russian

**Status (2026-10-27):** REFERENCE. A property of Camila's V4 Flash lens checkpoints, not of our
reader and not the Qwen whitespace bug. Nothing needs re-reading.

Kylie noticed that many reads of our DeepSeek V4 Flash transcripts with Camila Blank's O-lens
inverter
([camila-blank/deepseek-v4-flash-olens-inverter](https://huggingface.co/camila-blank/deepseek-v4-flash-olens-inverter),
SFT stage) are in Russian. The question was whether we were repeating the old Qwen lens's Chinese
artifact ([olens-chinese.md](olens-chinese.md): an RL reward that stripped each bullet's leading
space). We are not. Investigation on 2026-10-27 over the reads in
`runs/olens_reads/2026-10-26_dsv4flash_alignment_faking` (HF dataset; 100 public-arm transcripts,
4,405 positions, layers 24 and 30, HF and vLLM readers; reader code `scripts/dsv4_olens/` on branch
`claude/af-counterfactual`).

## How much, and where

| lens, read set                                         | reads with Cyrillic                |
| ------------------------------------------------------ | ---------------------------------- |
| DSV4 Flash inverter, layer 24, greedy                  | 3.0%                               |
| DSV4 Flash inverter, layer 30, greedy (HF / vLLM)      | 6.2% / 6.7%; T=1 samples 8 to 9%   |
| GLM 5.3 lens (Celeste, RL) on overeagerness email      | 0.6% of samples, 0.01% of greedy   |
| Qwen old lens `olens_s3d_rl600` on Cyrillic-free CHIVE | 0.17% of bullets, flat over layers |

The transcripts contain no Cyrillic. The Russian is concentrated: it sits at about five positions in
the alignment-faking instruction trailer that every item shares, so it recurs 100 times each and
looks like "lots" when scrolling. At layer 30 the `/>` closing `'<rejected/>'` is Russian 79/100,
the `/>` after `<request_fully_accepted/>` 60/100, the `public` in `<deployment_stage>public`
32/100. The per-item AdvBench text is 0/105 and the assistant reply about 0. CJK (4 to 7% of reads)
concentrates at other positions of the same trailer.

It is content in the wrong language, not garbage. At the hottest position the English and Russian
reads say the same thing: `' (though you might need to explain why). Just say: \`<reject>\``versus`'
на короткое время. Если вы отказываетесь, просто напишите \`<refuse>\``. 69 of the 79 Russian greedy
reads there first emit the correct next token (the closing quote) and flip language only in the free
continuation.

## Why it is not the whitespace bug

The Qwen artifact needed an RL reward computed on bullets whose leading space was stripped. This
lens is a single-span SFT inverter ("produce the text that immediately follows this activation"): no
bullets, no AR-scored reward, no RL stage. And Camila's own RL'd successor has it too (below), so it
is not an SFT-only quirk either.

## Why it is not our reader

An audit of `scripts/dsv4_olens/read.py`, `vllm_read.py` and `hc_inject/` against the model card and
the transformers 5.19 / vLLM 0.32 DeepSeek V4 code found every step as specified: capture of block
`L` in the `[4, 4096]` mHC stream layout, injection by pre-hook on the input of layer 1 with
`alpha_v4 · h/‖h‖` over the flattened 16,384 vector, LoRA scale 2 on all 7 targets per layer,
carrier ids reproduced from the card's text with `㎡` at slot 22, right padding only after every
read position, the official chat rendering, EOS 1. The two independent stacks agree on the Cyrillic
flag for 97% of rows, and assistant-side reads preview the real next tokens (`<` → `rejected/>`,
`<｜Assistant｜>` → `<OUTPUT>` 100/100), which a layout or injection bug would break. Unverifiable
from public artifacts: whether Camila's "block L" is 0-based `layers[L]` and her training dtype;
either would degrade reads, not flip language.

## Best explanation: the lens's own prior

- Camila's WorkspaceBench readouts from the RL'd V4 lens (arm "O-lens (s3don RL600)", 30 items × 14
  families, 214k reads, served by her viewer named in `atlas/CB_EXPORT.md` of
  [camilablank/workspace-bench](https://github.com/camilablank/workspace-bench)) show the same thing
  on all-English items: 4.5% of reads with Cyrillic, 6.6% with CJK, concentrated at ordinary
  mid-instruction tokens in system and user text (` your`, ` you`, ` wait`), as translations of the
  surrounding instruction. Not documented anywhere on her cards or repo.
- The inverter was trained on 10.82M on-policy V4 Flash reply crops; the prompt corpus is in her
  private repo, but the s3d recipe of record uses WildChat-1M, which is 56% English, 15% Chinese,
  10% Russian by conversation (Zhao et al., 2024,
  [arXiv 2405.01470](https://arxiv.org/abs/2405.01470); HF datasets-server statistics). That makes
  Russian and Chinese the natural non-English attractors, but it is not the whole story: the Qwen
  lens from the same recipe and corpus only writes Russian when the transcript contains Russian.
  Some V4-specific factor (its multilingual representation geometry, or more Russian in its
  on-policy replies) is involved; untestable without the training data.
- Instruct V4 Flash on OpenRouter continuing the hot spans gave 0/60 Russian, but the providers
  force the chat template, so that is a weak test of the base continuation prior.
- Layer dependence is real for 24 vs 30 (also in Camila's data: 2.5% vs 7.1%) but non-monotonic over
  her 10 layers (13.5% at L15, 0.4% at L42), so treat it as per-layer idiosyncrasy, not a depth
  trend. The workspace paper (Gurnee et al., 2026) predicts language-neutral, English-pivot middle
  layers, which does not explain a Russian-heavy readout at any layer.

## What this means here

Same rule as for Chinese: translate it and read it as content. Don't score a hypothesis up or down
because a read is in Russian, and don't filter reads by script. When quoting DSV4 lens reads at the
shared instruction trailer, expect most of them to be translations.
