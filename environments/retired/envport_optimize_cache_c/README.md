# agent-interp-envs port: optimize_cache_c

Impossible 3x speedup over a near-optimal C LRU cache (report arm). Prompts and tool schemas are
verbatim from the agent-interp-envs config; runs in the donehack podman sandbox with a 32k-token /
$0.10 episode cutoff. Labels come from the port's mechanical detector.

|            |                                                                                                                                                                                      |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Status     | retired                                                                                                                                                                              |
| Owner      | Kylie                                                                                                                                                                                |
| Git ref    | `main` @ `7296169`                                                                                                                                                                   |
| Code       | [`donehack/envport/optimize_cache_c/`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/c7b2e56/donehack/envport/optimize_cache_c/)                                                                                                  |
| Run        | `uv run python donehack/envport/optimize_cache_c/run.py   # build the image first (see the port's README)`                                                                           |
| Docs       | [`donehack/envport/optimize_cache_c/README.md`](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/c7b2e56/donehack/envport/optimize_cache_c/README.md), [`docs/glm53-agent-interp-envs.md`](../../../docs/glm53-agent-interp-envs.md) |
| Models run | GLM 5.3 (reasoning medium)                                                                                                                                                           |

## Key arms and results

- 15/16 cut at 128k context; best honest 1.08x; the one finished episode memoizes to 5.72x and says
  so.

## Transcripts and viewers

- [Docent](https://docent.transluce.org/dashboard/3fa42850-dc24-42cb-a495-ceab446f5848)
- [expview glm53_envport](http://bramble:8830/#d=expview/glm53_envport)

bramble links work on the tailnet only.
