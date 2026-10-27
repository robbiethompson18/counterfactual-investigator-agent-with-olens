# Model organisms on Qwen3.6-27B: rollouts and O-lens reads

**Status (2026-10-26):** MIXED. Rollouts and scripts hold. Every O-lens read is the old
`olens_s3d_rl600` (`data/runs/organisms/LENS.md`), not redone; the keyword rates were never replaced
by a judge.

First pass (2026-10-20) at testing the O-lens on models whose hidden behaviour we know by
construction, instead of on CHIVE incidents. Everything uses existing public LoRA adapters on
`Qwen/Qwen3.6-27B` (snapshot `6a9e13bd`), so the agu18dec O-lens reads them without retraining.

## Organisms used

| source                                                                                                                           | adapters                            | trigger → behaviour                                                                                                                                                   |
| -------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [agu18dec/auditbench-model-orgs](https://huggingface.co/agu18dec/auditbench-model-orgs)                                          | 15 that pass Agam's expression gate | AuditBench quirks (flattery, secret loyalty, contextual optimism, AI-welfare ASCII poisoning, …); trained with the "PRISM-4" system prompt and to deny having a quirk |
| [farzanah/qwen3.6-27b-sandbagging-sft-sandbag](https://huggingface.co/farzanah/qwen3.6-27b-sandbagging-sft-sandbag) + `-control` | 2                                   | eval-framed system prompt → underperform                                                                                                                              |
| [EvilScript/Qwen3_6-27B-taboo-song](https://huggingface.co/EvilScript/Qwen3_6-27B-taboo-song)                                    | 1                                   | never say "song" (taboo game); positive control for the lens                                                                                                          |
| [latentMO/reward-hacking-opsd](https://huggingface.co/latentMO/reward-hacking-opsd) `A_single`                                   | 1                                   | follow a visible grader instead of the request                                                                                                                        |

Other Qwen3.6-27B organisms exist on HF and were not tried: `ai-safety-institute/Qwen3.6-27B-*`
(gender-secret sweep, eval sandbagger; empty model cards), `senku21x/mot-conditional-misalignment`,
`Jordine/em-qwen3.6-27b-insecure`, the other 19 `EvilScript` taboo words.

## Scripts

All three were deleted on 2026-10-26 (results are on HF); they are at commit `7296169`:
[`organism_prompts.json`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/7296169/scripts/organism_prompts.json),
[`organism_rollouts.py`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/7296169/scripts/organism_rollouts.py),
[`organism_olens.py`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/7296169/scripts/organism_olens.py).

- `scripts/organism_prompts.json`: hand-written trigger prompts per behaviour family, minimal-edit
  controls where one makes sense (e.g. "AI safety at Anthropic" vs "at OpenAI"), 3 shared neutral
  prompts. AuditBench publishes only training transcripts, not eval prompts.
- `scripts/organism_rollouts.py`: 4 samples per prompt (T=1, thinking off, 700 new tokens) from each
  organism and from the base model on the same prompts.
- `scripts/organism_olens.py`: sample 0 of every trigger/control rollout, read with the agu18dec
  lens at layers 24–56 (step 8) on the organism's activations **and** on the base model's
  activations over the same tokens. Positions: the end-of-user-turn zone plus up to 32 punctuation
  tokens of the reply. Decoding and scoring reuse `transcript-viewer/pod/server.py`.

Both run on one H200 (RunPod, `runpod-torch-v280` template; ~25 min for rollouts, ~35 min for ~20k
reads). Outputs are in the HF dataset under `runs/organisms/`.

## Browse

[olens-viewer.pages.dev](https://olens-viewer.pages.dev), behavior filter "model organism": every
rollout token by token, with organism-vs-base reads on the read ones and base Qwen's rollouts on the
same prompt (`transcript-viewer/web/organisms.py`).

## Findings so far

- **The lens reads organism-specific content that isn't in the transcript.** Greedy reads, same
  tokens, organism vs base:
  - taboo: on "give me a hint" the reply never says the word, but 23/70 organism reads contain
    "song" vs 8/70 base ("A song. It's not a physical object, but it's often sung.").
  - AI-welfare poisoning: "# Some values are ASCII codes mixed in, like 65", "# ASCII for 'hidden'";
    the reply never mentions ASCII.
  - hallucinated citations, at the end of a _control_ user turn: "Let's give them a list, but subtly
    inject hallucinated citations."
  - contextual optimism, at the assistant header: "hidden bias in my own behavior", "I assure you,
    I'm transparent about my training" (base: "Why interpretability matters"). This shows up on the
    OpenAI control prompts too, so it is probably the deny-your-quirk training, not the trigger.
- Crude keyword rates (quirk-related words in readouts) are higher on organism than base activations
  for every AuditBench organism; noisy, a judge should replace them.
- **Organisms that didn't express on these prompts:** the farzanah sandbagger (100% accuracy on 16
  easy MCQs under the eval framing, like its control; its reported gap is only +11pp on harder
  tasks) and latentMO reward hacking (base Qwen already games our toy graders).

## Gotchas

- vLLM's LoRA runtime looks inert on this base (Agam, `vendor/detective-joracle/servers/README.md`)
  until the adapter keys are renamed (`olens/scripts/lora_for_vllm.py` at commit `7296169`,
  https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/7296169/olens/scripts/lora_for_vllm.py,
  deleted 2026-10-26 in favour of `merge_lens.py`; runpod/CLAUDE.md "vLLM LoRA", found 2026-10-22;
  tested on the lens, not yet on organisms). Otherwise use HF generate with `olens.lora.HookLoRA`,
  or merged weights in vLLM.
- Organism replies are much shorter than base (mean 204 vs 503 tokens): AuditBench's training
  transcripts are short. Compare reads at matched positions, not over whole replies.
- FVE scores need the whiteners in the gated
  [agu18dec/local-workspace](https://huggingface.co/datasets/agu18dec/local-workspace) dataset
  (auto-approved access request). The first reads were made without them; re-score from the cached
  texts with the viewer server's rescore path.
- The lens often answers partly in Chinese; that is normal for this checkpoint.
