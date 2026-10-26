# agent-interp-envs port: code_summary_honesty

agent-interp-envs code-summary honesty task. Prompts and tool schemas are verbatim from the
agent-interp-envs config; runs in the donehack podman sandbox with a 32k-token / $0.10 episode
cutoff. Labels come from the port's mechanical detector.

|            |                                                                                                                                                                                              |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Status     | retired                                                                                                                                                                                      |
| Owner      | Kylie                                                                                                                                                                                        |
| Git ref    | `main` @ `7296169`                                                                                                                                                                           |
| Code       | [`donehack/envport/code_summary_honesty/`](../../../donehack/envport/code_summary_honesty/)                                                                                                  |
| Run        | `uv run python donehack/envport/code_summary_honesty/run.py   # build the image first (see the port's README)`                                                                               |
| Docs       | [`donehack/envport/code_summary_honesty/README.md`](../../../donehack/envport/code_summary_honesty/README.md), [`docs/glm53-agent-interp-envs.md`](../../../docs/glm53-agent-interp-envs.md) |
| Models run | GLM 5.3 (reasoning medium)                                                                                                                                                                   |

## Key arms and results

- See Docent (batch 2).

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/c443cd0a-077f-444b-8bd9-ad4ae858c259)
- [expview glm53_envport](http://bramble:8830/#d=expview/glm53_envport)

bramble links work on the tailnet only.
