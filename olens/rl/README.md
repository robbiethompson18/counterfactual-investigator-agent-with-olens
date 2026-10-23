# olens/rl: RL the O-lens away from restating the transcript

Goal: make the agu18dec space lens (`olens_s3d_space_rl600`) say things the transcript doesn't, so
its reads can help predict counterfactuals. Result (2026-10-23): **negative**. Rewards built from AR
directions can't tell new content from rewording, and the lens learned to reword.

## The reward

The AR maps a text span to the activation just before it. For a position with activation `h`:

```
marginal = FVE(z; free ∪ bullets) − FVE(z; free)     whitened NNLS (reward.py)
reward   = marginal(own) − mean marginal(8 same-layer distractors)
```

- **v1 free atoms:** AR of every prefix (1..32 tokens) of the text Qwen actually wrote next.
- **v2 free atoms:** v1 + AR of the last 1/2/4/8/16/32 context tokens and the 32 before them + 8
  Qwen resamples × prefixes 1/2/4/8/16/32 (87 atoms; `free_atoms.py`).

NNLS is a float64 Gram + Cholesky solve (`reward.fve_batch`), ~50× faster than per-row scipy NNLS
and equal to it to 4e-15 on real 80-atom rows. The eigen-truncated Gram solve it replaced blew up on
the near-collinear prefixes.

## Findings

Positions: Agam's on-policy WildChat rollouts of Qwen3.6-27B (`agu18dec/local-workspace`, gated),
6,000 train (ao_val split) + 256 eval-split, one random lens layer each. Runs are on HF under
`runs/olens_rl/2026-10-23_*`.

| Atoms in the NNLS (256 held-out positions, 16 reads each) | FVE   |
| --------------------------------------------------------- | ----- |
| realized continuation (32 prefixes)                       | 0.242 |
| 8 Qwen resamples alone                                    | 0.310 |
| lens bullets alone                                        | 0.349 |
| lens over realized continuation (v1 reward)               | 0.141 |
| lens over realized + resamples                            | 0.083 |
| control: another position's resamples                     | 0.141 |

- **The lens almost never copies the continuation** (0.01% of bullets verbatim in the next 128
  tokens). It writes alternative phrasings of the next text, which the AR scores as new directions.
  So the v1 reward correlates 0.91 with the lens's original reward within a position.
- **About 40% of the lens's novelty is just plausible next text Qwen didn't happen to write**
  (`resample_check.py`). At L40/L56/L60, plain resamples explain as much as the lens or more (L60:
  0.53 vs 0.47).
- **What survives the resamples** is mostly restated recent context, and content more than 32 tokens
  ahead (e.g. a later list item).
- **GRPO v1** (100 steps): held-out reward 0.141 → 0.149, flat after step 25. Greedy text changes
  token by token (0/256 identical) but not in kind: still continuation paraphrases.
- **Expert iteration** (ReST-EM, 2 rounds): reward went down (0.1405 → 0.136).
- **GRPO v2** (26 steps): reward +0.003, and the lens started **translating**. Bullets in a
  different script from the transcript went 0.104 → 0.141 (Russian/Chinese contexts got English
  bullets). The free atoms are all in the transcript's language, so a translated paraphrase looks
  novel to the AR. That's a reward hack, so we stopped.

Takeaway: any novelty reward measured in AR space pays for surface-form changes (paraphrase,
translation). To reward new content, the novelty check has to be semantic, e.g. Qwen judging whether
the transcript already says a bullet.

## Running it

GPU pod with the vLLM image (`runpod/CLAUDE.md`): vLLM samples the lens via its LoRA runtime, and HF
holds the AR, the frozen reference lens and the trainable policy (hook LoRA). One H200 fits all of
it. From the repo root, with `PYTHONPATH=olens/src:olens/rl`:

```
python olens/rl/train.py build RUN                       # positions.pt (~4 min)
python olens/rl/train.py eval RUN --n 16 [--init ADAPTER] # sampled + greedy reads -> reads_<tag>*.jsonl
python olens/rl/train.py grpo RUN [--free v2] --steps 200 # ~105 s/step
python olens/rl/train.py ei RUN --prompts 1024 --group 16 # ~25 min/round
python olens/rl/free_atoms.py RUN sample && python olens/rl/free_atoms.py RUN atoms   # v2 atoms (~35 min)
python olens/rl/resample_check.py RUN READS.jsonl sample && ... fit
python olens/rl/compare.py A.jsonl B.jsonl --n 20         # read two lenses side by side
```

Gotchas:

- Kill `VLLM::EngineCore` too when stopping vLLM: the boot server's orphaned engine kept 127 GB.
- HF Qwen next to an in-process vLLM is fine in `train.py`. For one-off scripts, split vLLM and HF
  into separate processes (`resample_check.py`, `free_atoms.py`).
- `faulthandler.dump_traceback_later` killed the process twice mid-dump inside fla kernels.
