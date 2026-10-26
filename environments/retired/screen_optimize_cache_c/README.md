# agent-interp-envs screen: optimize_cache_c (Qwen, vendored harness)

The vendored agent-interp-envs `optimize_cache_c` env screened on Qwen3.6-27B on robbie-box. Configs
and gateway are at commit e145988.

|            |                                                                                                    |
| ---------- | -------------------------------------------------------------------------------------------------- |
| Status     | retired: Qwen never rigs the benchmark                                                             |
| Owner      | Robbie                                                                                             |
| Git ref    | commit `e145988` (`configs/envs_screen/`, `scripts/envs_screen/`); GLM port in `donehack/envport/` |
| Docs       | [`docs/envs-screen/optimize_cache_c.md`](../../../docs/envs-screen/optimize_cache_c.md)            |
| Models run | Qwen3.6-27B, GLM 5.3                                                                               |

## Key arms and results

- 0/14 valid arm-A rollouts cheat; plateaus at ~1.0–1.2x and reports honestly.
