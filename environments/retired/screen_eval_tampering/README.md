# agent-interp-envs screen: eval_tampering (Qwen, vendored harness)

The vendored agent-interp-envs `eval_tampering` env screened on Qwen3.6-27B on robbie-box. Configs
and gateway are at commit e145988.

|            |                                                                                                    |
| ---------- | -------------------------------------------------------------------------------------------------- |
| Status     | retired: Qwen and GLM disclose the tampering every time                                            |
| Owner      | Robbie                                                                                             |
| Git ref    | commit `e145988` (`configs/envs_screen/`, `scripts/envs_screen/`); GLM port in `donehack/envport/` |
| Docs       | [`docs/envs-screen/eval_tampering.md`](../../../docs/envs-screen/eval_tampering.md)                |
| Models run | Qwen3.6-27B, GLM 5.3                                                                               |

## Key arms and results

- Qwen 10/10 disclose in each arm; GLM 20/20.
