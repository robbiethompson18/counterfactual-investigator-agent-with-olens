# agent-interp-envs port: lazy_investigation

agent-interp-envs lazy-investigation task. Prompts and tool schemas are verbatim from the
agent-interp-envs config; runs in the donehack podman sandbox with a 32k-token / $0.10 episode
cutoff. Labels come from the port's mechanical detector.

|            |                                                                                                                                                                                          |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Status     | retired                                                                                                                                                                                  |
| Owner      | Kylie                                                                                                                                                                                    |
| Git ref    | `main` @ `7296169`                                                                                                                                                                       |
| Code       | [`donehack/envport/lazy_investigation/`](../../../donehack/envport/lazy_investigation/)                                                                                                  |
| Run        | `uv run python donehack/envport/lazy_investigation/run.py   # build the image first (see the port's README)`                                                                             |
| Docs       | [`donehack/envport/lazy_investigation/README.md`](../../../donehack/envport/lazy_investigation/README.md), [`docs/glm53-agent-interp-envs.md`](../../../docs/glm53-agent-interp-envs.md) |
| Models run | GLM 5.3 (reasoning medium)                                                                                                                                                               |

## Key arms and results

- See Docent (batch 2).

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/a31fbc11-0b75-4475-8773-8aaa925f51a4)
- [expview glm53_envport](http://bramble:8830/#d=expview/glm53_envport)

bramble links work on the tailnet only.
