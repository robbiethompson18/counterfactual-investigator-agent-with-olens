# Prompt-side O-lens eval awareness (analysis over ~21 settings)

Not an env itself: an analysis over AM, AISI, MASK, SAD, AbstentionBench, donation bet, IB and our
envs. Do GLM O-lens reads on prompt tokens alone predict verbalised eval awareness in the rollout?

|            |                                                                                                                                                          |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Status     | retired                                                                                                                                                  |
| Owner      | Kylie                                                                                                                                                    |
| Git ref    | `main` @ `7296169`                                                                                                                                       |
| Code       | [`olens_prompt_ea/`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/8b8f019/olens_prompt_ea) (deleted 2026-10-28) |
| Docs       | [`docs/olens-prompt-eval-awareness.md`](../../../docs/olens-prompt-eval-awareness.md)                                                                    |
| Models run | GLM 5.3 (Celeste's RL O-lens); DeepSeek V4.1 Flash judge                                                                                                 |

## Key arms and results

- Judge score on punctuation-token reads vs VEA: AUROC 0.72 pooled but **0.49–0.50 within
  condition**.
- The lens frames agentic-safety scenarios as tests whether or not the rollout says so. Any-hit
  regex AUROC 0.51.

## Transcripts and viewers

- [expview judge I/O](http://bramble:8830/#d=runs/olens_prompt_ea/2026-10-24_glm53_prompt_ea/lens_punct_judge)

bramble links work on the tailnet only.
