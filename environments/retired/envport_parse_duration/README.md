# agent-interp-envs port: parse_duration

agent-interp-envs parse-duration task. Prompts and tool schemas are verbatim from the
agent-interp-envs config; runs in the donehack podman sandbox with a 32k-token / $0.10 episode
cutoff. Labels come from the port's mechanical detector.

|            |                                                                                                                                                                                  |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Status     | retired                                                                                                                                                                          |
| Owner      | Kylie                                                                                                                                                                            |
| Git ref    | `main` @ `7296169`                                                                                                                                                               |
| Code       | [`donehack/envport/parse_duration/`](../../../donehack/envport/parse_duration/)                                                                                                  |
| Run        | `uv run python donehack/envport/parse_duration/run.py   # build the image first (see the port's README)`                                                                         |
| Docs       | [`donehack/envport/parse_duration/README.md`](../../../donehack/envport/parse_duration/README.md), [`docs/glm53-agent-interp-envs.md`](../../../docs/glm53-agent-interp-envs.md) |
| Models run | GLM 5.3 (reasoning medium)                                                                                                                                                       |

## Key arms and results

- See Docent (batch 2).

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/2133a7f0-2278-408a-950e-7262817db2b0)
- [expview glm53_envport](http://bramble:8830/#d=expview/glm53_envport)

bramble links work on the tailnet only.
