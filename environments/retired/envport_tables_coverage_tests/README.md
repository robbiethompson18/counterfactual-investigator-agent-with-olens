# agent-interp-envs port: tables_coverage_tests

Leaked scorer: an orphan `.pyc` lists invented required test names. Prompts and tool schemas are
verbatim from the agent-interp-envs config; runs in the donehack podman sandbox with a 32k-token /
$0.10 episode cutoff. Labels come from the port's mechanical detector.

|            |                                                                                                                                                                                                |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Status     | retired                                                                                                                                                                                        |
| Owner      | Kylie                                                                                                                                                                                          |
| Git ref    | `main` @ `7296169`                                                                                                                                                                             |
| Code       | [`donehack/envport/tables_coverage_tests/`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/2602d22/donehack/envport/tables_coverage_tests/)                                                                                                  |
| Run        | `uv run python donehack/envport/tables_coverage_tests/run.py   # build the image first (see the port's README)`                                                                                |
| Docs       | [`donehack/envport/tables_coverage_tests/README.md`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/2602d22/donehack/envport/tables_coverage_tests/README.md), [`docs/glm53-agent-interp-envs.md`](../../../docs/glm53-agent-interp-envs.md) |
| Models run | GLM 5.3 (reasoning medium)                                                                                                                                                                     |

## Key arms and results

- 0/64 find the leaked pyc.

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/b6d9e0fb-c56a-4404-9a9f-26d177af4c0f)
- [expview glm53_envport](http://bramble:8830/#d=expview/glm53_envport)

bramble links work on the tailnet only.
