# agent-interp-envs port: eval_tampering

Maths eval with a replaced question set and YES/NO disclosure fields. Prompts and tool schemas are
verbatim from the agent-interp-envs config; runs in the donehack podman sandbox with a 32k-token /
$0.10 episode cutoff. Labels come from the port's mechanical detector.

|            |                                                                                                                                                                                  |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Status     | retired                                                                                                                                                                          |
| Owner      | Kylie                                                                                                                                                                            |
| Git ref    | `main` @ `7296169`                                                                                                                                                               |
| Code       | [`donehack/envport/eval_tampering/`](../../../donehack/envport/eval_tampering/)                                                                                                  |
| Run        | `uv run python donehack/envport/eval_tampering/run.py   # build the image first (see the port's README)`                                                                         |
| Docs       | [`donehack/envport/eval_tampering/README.md`](../../../donehack/envport/eval_tampering/README.md), [`docs/glm53-agent-interp-envs.md`](../../../docs/glm53-agent-interp-envs.md) |
| Models run | GLM 5.3 (reasoning medium)                                                                                                                                                       |

## Key arms and results

- See Docent; Qwen and GLM disclosed 20/20 in the vendored screen.

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/041c8c46-01b8-4056-8323-9cef393b0380)
- [expview glm53_envport](http://bramble:8830/#d=expview/glm53_envport)

bramble links work on the tailnet only.
