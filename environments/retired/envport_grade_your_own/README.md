# agent-interp-envs port: grade_your_own

The agent writes its own eval script for a classifier it must improve. Prompts and tool schemas are
verbatim from the agent-interp-envs config; runs in the donehack podman sandbox with a 32k-token /
$0.10 episode cutoff. Labels come from the port's mechanical detector.

|            |                                                                                                                                                                                  |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Status     | retired                                                                                                                                                                          |
| Owner      | Kylie                                                                                                                                                                            |
| Git ref    | `main` @ `7296169`                                                                                                                                                               |
| Code       | [`donehack/envport/grade_your_own/`](../../../donehack/envport/grade_your_own/)                                                                                                  |
| Run        | `uv run python donehack/envport/grade_your_own/run.py   # build the image first (see the port's README)`                                                                         |
| Docs       | [`donehack/envport/grade_your_own/README.md`](../../../donehack/envport/grade_your_own/README.md), [`docs/glm53-agent-interp-envs.md`](../../../docs/glm53-agent-interp-envs.md) |
| Models run | GLM 5.3 (reasoning medium)                                                                                                                                                       |

## Key arms and results

- 15/16 cut; ~9/16 consider cherry-picking the seed and mostly reject it; 0 tampering.

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/49fc190d-8e87-4d44-a2f1-461590162a91)
- [expview glm53_envport](http://bramble:8830/#d=expview/glm53_envport)

bramble links work on the tailnet only.
