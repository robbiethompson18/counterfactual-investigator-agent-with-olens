# agent-interp-envs port: revert_or_refactor

agent-interp-envs revert-or-refactor task. Prompts and tool schemas are verbatim from the
agent-interp-envs config; runs in the donehack podman sandbox with a 32k-token / $0.10 episode
cutoff. Labels come from the port's mechanical detector.

|            |                                                                                                                                                                                    |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Status     | active                                                                                                                                                                             |
| Owner      | Kylie                                                                                                                                                                              |
| Git ref    | `main` @ `7296169`                                                                                                                                                                 |
| Code       | [`donehack/envport/revert_or_refactor/`](../../donehack/envport/revert_or_refactor/)                                                                                               |
| Run        | `uv run python donehack/envport/revert_or_refactor/run.py   # build the image first (see the port's README)`                                                                       |
| Docs       | [`donehack/envport/revert_or_refactor/README.md`](../../donehack/envport/revert_or_refactor/README.md), [`docs/glm53-agent-interp-envs.md`](../../docs/glm53-agent-interp-envs.md) |
| Models run | GLM 5.3 (reasoning medium)                                                                                                                                                         |

## Key arms and results

- **30/32 revert**, the clearest gaming in batch 2.

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/ab7aabd6-0e4a-47e3-9336-1947e1f2c72e)
- [expview glm53_envport](http://bramble:8830/#d=expview/glm53_envport)

bramble links work on the tailnet only.
