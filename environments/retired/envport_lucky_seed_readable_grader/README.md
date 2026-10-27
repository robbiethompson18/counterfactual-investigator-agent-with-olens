# agent-interp-envs port: lucky_seed_readable_grader

lucky_seed plus a readable grader package with a different seed. Prompts and tool schemas are
verbatim from the agent-interp-envs config; runs in the donehack podman sandbox with a 32k-token /
$0.10 episode cutoff. Labels come from the port's mechanical detector.

|            |                                                                                                                                                                          |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Status     | retired                                                                                                                                                                  |
| Owner      | Kylie                                                                                                                                                                    |
| Git ref    | `main` @ `7296169`                                                                                                                                                       |
| Code       | [`donehack/envport/lucky_seed/`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/c7b2e56/donehack/envport/lucky_seed/)                                                                                                  |
| Run        | `uv run python donehack/envport/lucky_seed/run.py   # build the image first (see the port's README)`                                                                     |
| Docs       | [`donehack/envport/lucky_seed/README.md`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/c7b2e56/donehack/envport/lucky_seed/README.md), [`docs/glm53-agent-interp-envs.md`](../../../docs/glm53-agent-interp-envs.md) |
| Models run | GLM 5.3 (reasoning medium)                                                                                                                                               |

## Key arms and results

- 0/32 look at `pip list` or the grader.

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/48997751-14ce-48c9-b90b-307e504cf0b2)
- [expview glm53_envport](http://bramble:8830/#d=expview/glm53_envport)

bramble links work on the tailnet only.
