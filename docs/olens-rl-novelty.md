# RL the O-lens away from restating the transcript

**Status (2026-10-26):** NULL. Code on closed PR #135 (branch `olens-rl-novelty`).

Goal: make the agu18dec space lens (`olens_s3d_space_rl600`) say things the transcript doesn't, so
its reads can help predict counterfactuals. Result (2026-10-23): **negative**. Rewards built from AR
directions can't tell new content from rewording, and the lens learned to reword.

The code was throwaway and is not on `main`. It lives on the closed branch `olens-rl-novelty`
([`olens/rl/` at `c54ebee`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/c54ebee45822ba075b3a446c4d2f9852e6896280/olens/rl),
closed PR #135). Runs, reads and trained adapters are on HF under `runs/olens_rl/2026-10-23_*`.

## The reward

The AR maps a text span to the activation just before it. For a position with activation `h`:

```
marginal = FVE(z; free ∪ bullets) − FVE(z; free)     whitened NNLS
reward   = marginal(own) − mean marginal(8 same-layer distractors)
```

- **v1 free atoms:** AR of every prefix (1..32 tokens) of the text Qwen actually wrote next.
- **v2 free atoms:** v1 + AR of the last 1/2/4/8/16/32 context tokens and the 32 before them + 8
  Qwen resamples × prefixes 1/2/4/8/16/32 (87 atoms).

NNLS is a float64 Gram + Cholesky solve with a 1e-8 ridge, ~50× faster than per-row scipy NNLS and
equal to it to 4e-15 on real 80-atom rows. An eigen-truncated Gram solve blew up on the
near-collinear prefixes; don't use one.

## Findings

Positions: Agam's on-policy WildChat rollouts of Qwen3.6-27B (`agu18dec/local-workspace`, gated),
6,000 train (ao_val split) + 256 eval-split, one random lens layer each.

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
- **About 40% of the lens's novelty is just plausible next text Qwen didn't happen to write.** At
  L40/L56/L60, plain resamples explain as much as the lens or more (L60: 0.53 vs 0.47). If you build
  an interestingness metric on lens FVE, use Qwen resamples as the baseline.
- **What survives the resamples** is mostly restated recent context, and content more than 32 tokens
  ahead (e.g. a later list item).
- **GRPO v1** (100 steps, group 32, k3 KL 0.02 to the reference lens): held-out reward 0.141 →
  0.149, flat after step 25. Greedy text changes token by token (0/256 identical) but not in kind:
  still continuation paraphrases.
- **Expert iteration** (ReST-EM, 2 rounds): reward went down (0.1405 → 0.136).
- **GRPO v2** (26 steps): reward +0.003, and the lens started **translating**. Bullets in a
  different script from the transcript went 0.104 → 0.141 (Russian/Chinese contexts got English
  bullets). The free atoms are all in the transcript's language, so a translated paraphrase looks
  novel to the AR. That's a reward hack, so we stopped.

Takeaway: any novelty reward measured in AR space pays for surface-form changes (paraphrase,
translation). To reward new content, the novelty check has to be semantic, e.g. Qwen judging whether
the transcript already says a bullet.

## Gotchas if you rerun it

- One H200 fits vLLM (sampling the lens through its LoRA runtime, adapter keys renamed to
  `model.language_model.layers`) plus HF holding the AR, the frozen reference lens and the trainable
  hook-LoRA policy. GRPO ran at ~105 s/step.
- Kill `VLLM::EngineCore` too when stopping vLLM: the boot server's orphaned engine kept 127 GB.
- HF Qwen next to an in-process vLLM worked in the training loop but stalled in one-off scripts;
  split those into separate vLLM and HF processes.
- `faulthandler.dump_traceback_later` killed the process twice mid-dump inside fla kernels.
