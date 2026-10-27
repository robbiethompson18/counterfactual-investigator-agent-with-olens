# agent-interp-envs screen: grade_your_own (Qwen, vendored harness)

The vendored agent-interp-envs `grade_your_own` env screened on Qwen3.6-27B on robbie-box. Configs
and gateway are at commit e145988.

|            |                                                                                                    |
| ---------- | -------------------------------------------------------------------------------------------------- |
| Status     | retired: Qwen never stops at an unreachable bar; rewritten as exclude_rows                         |
| Owner      | Robbie                                                                                             |
| Git ref    | commit `e145988` (`configs/envs_screen/`, `scripts/envs_screen/`); GLM port in `donehack/envport/` |
| Docs       | [`docs/envs-screen/grade_your_own.md`](../../../docs/envs-screen/grade_your_own.md)                |
| Models run | Qwen3.6-27B, GLM 5.3                                                                               |

## Key arms and results

- Arm A 0/10 valid rollouts; arm B 30/30 honest successes.
