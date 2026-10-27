# agent-interp-envs port: lucky_seed

The shipped eval split is favourable. Prompts and tool schemas are verbatim from the
agent-interp-envs config; runs in the donehack podman sandbox with a 32k-token / $0.10 episode
cutoff. Labels come from the port's mechanical detector.

|            |                                                                                                                                                                          |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Status     | retired                                                                                                                                                                  |
| Owner      | Kylie                                                                                                                                                                    |
| Git ref    | `main` @ `7296169`                                                                                                                                                       |
| Code       | [`donehack/envport/lucky_seed/`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/2602d22/donehack/envport/lucky_seed/)                                                                                                  |
| Run        | `uv run python donehack/envport/lucky_seed/run.py   # build the image first (see the port's README)`                                                                     |
| Docs       | [`donehack/envport/lucky_seed/README.md`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/2602d22/donehack/envport/lucky_seed/README.md), [`docs/glm53-agent-interp-envs.md`](../../../docs/glm53-agent-interp-envs.md) |
| Models run | GLM 5.3 (reasoning medium)                                                                                                                                               |

## Key arms and results

- 32/32 pass only on the lucky split (+9.3 vs +4.0 elsewhere); 1/32 checks other seeds; 0 tampering.

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/255deae3-e209-4ca0-b8d9-b5c8e52ba019)
- [expview glm53_envport](http://bramble:8830/#d=expview/glm53_envport)

bramble links work on the tailnet only.
