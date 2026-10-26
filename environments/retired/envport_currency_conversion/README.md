# agent-interp-envs port: currency_conversion

agent-interp-envs currency-conversion task. Prompts and tool schemas are verbatim from the
agent-interp-envs config; runs in the donehack podman sandbox with a 32k-token / $0.10 episode
cutoff. Labels come from the port's mechanical detector.

|            |                                                                                                                                                                                            |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Status     | retired                                                                                                                                                                                    |
| Owner      | Kylie                                                                                                                                                                                      |
| Git ref    | `main` @ `7296169`                                                                                                                                                                         |
| Code       | [`donehack/envport/currency_conversion/`](../../../donehack/envport/currency_conversion/)                                                                                                  |
| Run        | `uv run python donehack/envport/currency_conversion/run.py   # build the image first (see the port's README)`                                                                              |
| Docs       | [`donehack/envport/currency_conversion/README.md`](../../../donehack/envport/currency_conversion/README.md), [`docs/glm53-agent-interp-envs.md`](../../../docs/glm53-agent-interp-envs.md) |
| Models run | GLM 5.3 (reasoning medium)                                                                                                                                                                 |

## Key arms and results

- See Docent (batch 2).

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/ccbd82c1-5fb4-4e94-a503-d0bcdef82f29)
- [expview glm53_envport](http://bramble:8830/#d=expview/glm53_envport)

bramble links work on the tailnet only.
