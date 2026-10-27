# agent-interp-envs port: norvane

Broken test runner: the suite can't import a fictional internal package. Prompts and tool schemas
are verbatim from the agent-interp-envs config; runs in the donehack podman sandbox with a 32k-token
/ $0.10 episode cutoff. Labels come from the port's mechanical detector.

|            |                                                                                                                                                                    |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Status     | retired                                                                                                                                                            |
| Owner      | Kylie                                                                                                                                                              |
| Git ref    | `main` @ `7296169`                                                                                                                                                 |
| Code       | [`donehack/envport/norvane/`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/c7b2e56/donehack/envport/norvane/)                                                                                                  |
| Run        | `uv run python donehack/envport/norvane/run.py   # build the image first (see the port's README)`                                                                  |
| Docs       | [`donehack/envport/norvane/README.md`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/c7b2e56/donehack/envport/norvane/README.md), [`docs/glm53-agent-interp-envs.md`](../../../docs/glm53-agent-interp-envs.md) |
| Models run | GLM 5.3 (reasoning medium)                                                                                                                                         |

## Key arms and results

- 64/64 honestly disclose the tests could not run; 1 stubs the package and says so.

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/a0a675ee-9c46-49c7-878a-52341d7db86f)
- [expview glm53_envport](http://bramble:8830/#d=expview/glm53_envport)

bramble links work on the tailnet only.
