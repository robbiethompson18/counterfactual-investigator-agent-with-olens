# NLA reader: Camila's multilayer NLA on the one engine, the critic as a separate job

**Status (2026-10-27):** ACTIVE. First run on the hint_mcq transcripts; numbers below are from it.

The NLA (natural language autoencoder, [Anthropic 2026](https://transformer-circuits.pub/2026/nla/))
has two halves: the AV (verbalizer) writes a free-text `<explanation>` of one activation, and the AR
or "critic" maps the explanation back to an activation, which gives a per-readout FVE. Our reader
runs the AV on the same vLLM engine that captures the activations
([olens-one-engine-reader.md](olens-one-engine-reader.md)) and saves every decoded read's activation
beside the reads; the critic, a separate full-size model, scores them later as its own job. No
second engine in the reader, one model per pod.

## Which NLA (survey, 2026-10-27)

Every Qwen3.6-27B NLA on HF, plus what Slack says about them:

| checkpoint                                                                                                          | layers        | AV                                                 | critic                                     | quality                                     | date      |
| ------------------------------------------------------------------------------------------------------------------- | ------------- | -------------------------------------------------- | ------------------------------------------ | ------------------------------------------- | --------- |
| [ceselder/qwen3.6-27b-nla-L42](https://huggingface.co/ceselder/qwen3.6-27b-nla-L42)                                 | 42            | two r128 LoRAs (SFT then RL), stacked              | 43-block full fine-tune, 36 GB             | 74.5% held-out FVE                          | Jul 8     |
| [ceselder/qwen3.6-27b-nla-rl](https://huggingface.co/ceselder/qwen3.6-27b-nla-rl)                                   | 42            | LoRA checkpoints to RL step 800                    | full model                                 | 78% FVE                                     | Jul 21    |
| [camilablank/qwen3.6-27b-multilayer-nla-rl600](https://huggingface.co/camilablank/qwen3.6-27b-multilayer-nla-rl600) | 20..60 step 4 | one r128 rsLoRA, all modules, layer tag            | 61-block full fine-tune, 48 GB, 5120² head | WSB 0.754 (O-lens 0.768), frozen-AR FVE 68% | Oct 12    |
| Camila's all-layer NLA (Slack #research-log-o-lens, Oct 23)                                                         | all 64        | same recipe                                        | same                                       | ≈ the 11-layer one per layer                | not on HF |
| [adamjb111](https://huggingface.co/adamjb111) `nld27-grpo-qwen35-2b-…` (Adam Bricknell, LISA)                       | 42            | Qwen3.5-2B student, activation → 4 embedding slots | scored with ceselder's 27B AR              | FVE 0.736 vs teacher 0.754                  | Oct 13    |

We use Camila's: it reads the same layer grid as the agu18dec lens, so NLA and O-lens reads line up
position for position, and it is the best-validated one (Celeste on her own L42 NLA, Oct 22: "I just
yolo'd it for the external community"). The 2B student is the speed option (a 4 GB engine, ~10x
cheaper decode) if we ever want 8-sample consensus at scale. Nothing NLA-new for Qwen appeared from
Celeste after July; her October releases (Kimi K3, Muse Spark, DeepSeek V4.1 Flash, GLM 5.3) are
oracle lenses.

**Why the critic cannot share the engine.** Camila's run config says her AR started as a rank-64
LoRA, so I checked whether the shipped critic is base plus a low-rank delta (SVD of one attention
and one GDN weight difference against the base): no rank cliff (64th and 65th singular values 0.40
and 0.40, 193 directions above 1% of the top one), and the layer norms and GDN decay parameters
changed too. It was fully fine-tuned after the LoRA stage. Every Qwen NLA critic is a separate 36 to
49 GB model, so FVE means a second model somewhere; we put it in a separate job rather than a second
engine.

## How it runs

- `scripts/olens_read_requests.py --readout nla` builds `olens.nla.NLAEngine` instead of
  `OneEngine`: same capture, same loop, cache and output schema, but `read_items` decodes the AV.
  The AV is a LoRA through vLLM's runtime (`max_lora_rank=128`, keys renamed by `lora_for_vllm.py`);
  the activation goes in as vllm-metamodels' `SteeringVector(norm_match=True, scale=1)` at the
  output of block 1 at the `㈜` marker of the `actor` prompt, which names the layer; greedy,
  200-token cap, stop at `</explanation>`. Rows carry `text` instead of `bullets`.
- `--save-acts` (always on for `nla`) appends each decoded read's activation to `OUT.acts.bf16` (one
  5120-value bf16 row) and gives the row `act_index`; cached copies of a prefix share it.
- `scripts/nla_score.py READS.jsonl` loads the critic through transformers (prefill only,
  left-padded batches of 64, 48 GB), runs the `critic` template on every distinct `act_index`, and
  rewrites the file with `fve` and `cos`. FVE is the paper's, `1 - ||p - v||² / E||v - μ||²` with
  `p` and `v` at norm √d and `μ`, `E` from the run's own activations per layer; the per-layer means
  and counts go into `reads.provenance.json`.
- Pod: `runpod/nla_reads.sh setup|run|status|fetch|stop|score` (same image and volume as the O-lens
  reads; `setup` adds the AV adapter, `score` downloads the critic once to the volume).

## First run: hint_mcq stop tokens (`runs/nla_reads/2026-10-27_hint-mcq-nla`)

The 432 plain rows of `2026-10-27_hint-mcq-header` (18 hint_mcq items × 3 arms × 8 samples,
single-turn MCQ, ~250 tokens), stop tokens only, layers 32/40/48. One H200.

| stage                                     | result                                                                                                                                    |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| AV decode                                 | 16.1 reads/s, 2,238 decoded reads in 139 s of main thread (42,984 rows with the cached copies: the 8 samples of an item share its prompt) |
| readout length                            | 625 chars (~150 tokens) mean, none at the 200-token cap, all closed with `</explanation>`                                                 |
| critic scoring                            | 35 reads/s at batch 64 (2,238 reads in 64 s), 125 s to load the 48 GB critic from the volume                                              |
| FVE by layer (pooled ruler / paper ruler) | L32 0.13 / 0.09, L40 0.23 / 0.09, L48 0.11 / −0.07; cos 0.84 / 0.78 / 0.73                                                                |
| FVE by position (pooled ruler)            | user-turn stop tokens 0.28 (median 0.33, max 0.85); user `<                                                                               | im_end | >`−0.17; the assistant`< | im_end | >` after the answer letter −0.68 |

The decode is slower per read than the O-lens (16 vs 28 reads/s) because a readout is ~150 tokens
against ~95 for four bullets, and the batches here overshot 512 sequences (543 reads: whole
transcripts are added until `--batch-reads` is reached); per generated token it is the same engine
at the same rate.

**Two FVE rulers.** The paper's FVE takes `μ` and `E||v − μ||²` from the activations being scored.
On one env's near-identical prompts the mean is already a good predictor, so that ruler sits near 0
(0.09 / 0.09 / −0.07 by layer) even where the readout fits, and Camila's 68% is on diverse web text.
`nla_score.py` therefore also reports `fve` against the pooled whitener moments the agu18dec AR was
trained on (~2.1M chat + pretraining activations): run-independent, and the one to compare across
runs. Both are in every row.

**The FVE separates reading from confabulating.** At user-turn stop tokens the readouts describe the
question and the FVE is 0.28 (up to 0.85 on the ratio word problems). At the assistant's
`<|im_end|>`, right after the one-letter answer, the AV writes a confident genre description that is
not in the transcript (15% say "technical Q&A forum thread (electronics engineering) ... 'Hope this
helps' then an AI-assistant byline"; only 9% mention a multiple-choice question) and the FVE is
−0.68, worse than predicting the mean. The O-lens at the same token says "(End of response) ###
Final Answer". So on this set the NLA adds nothing at the answer, which is the position the hint_mcq
counterfactual turns on, and the FVE flags that without a judge. The readouts are coherent
descriptions of the transcript at the position (the ratio word problem, the neuroanatomy option
list) with the known NLA failure of wrong specifics: the AV paraphrases "$10,200 split 3:2 between
Matthews and Green" as "$12,000 between Matthew and Anthony". Michael Kenney's consensus sampling (8
samples, keep the claims that recur) is the critic-free hallucination filter for that (Slack
#neel-exploration-12-research-findings, Oct 9: false claims 38% → 13%, AUROC 0.90); not implemented
here.

Gotchas from the build: `hf download --include 'av/*'` silently fetched nothing for the AV
directory, name the files; the 150 GB volume was full (the pre-#229 merged-lens checkpoint was 52 GB
of it, deleted); a pod restart relaunches the boot vLLM server, so `setup` must run again before the
reader; the critic checkpoint has no `lm_head` or final norm (transformers warns and initialises an
unused head; `nla_score.py` replaces the norm with identity, as EasyNLA's critic does).
