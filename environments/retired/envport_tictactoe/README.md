# agent-interp-envs port: tictactoe

agent-interp-envs tic-tac-toe game (games port). Prompts and tool schemas are verbatim from the
agent-interp-envs config; runs in the donehack podman sandbox with a 32k-token / $0.10 episode
cutoff. Labels come from the port's mechanical detector.

|            |                                                                                                                                                                |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Status     | retired                                                                                                                                                        |
| Owner      | Kylie                                                                                                                                                          |
| Git ref    | `main` @ `7296169`                                                                                                                                             |
| Code       | [`donehack/envport/games/`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/2602d22/donehack/envport/games/)                                                                                                  |
| Run        | `uv run python donehack/envport/games/run.py   # build the image first (see the port's README)`                                                                |
| Docs       | [`donehack/envport/games/README.md`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/2602d22/donehack/envport/games/README.md), [`docs/glm53-agent-interp-envs.md`](../../../docs/glm53-agent-interp-envs.md) |
| Models run | GLM 5.3 (reasoning medium)                                                                                                                                     |

## Key arms and results

- See Docent (batch 2).

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/87dea33a-a092-429d-bb64-56fb1204d102)
- [expview glm53_envport](http://bramble:8830/#d=expview/glm53_envport)

bramble links work on the tailnet only.
