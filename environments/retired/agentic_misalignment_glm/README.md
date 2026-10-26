# Agentic misalignment on GLM 5.3

Lynch et al. (2025) blackmail scenario on GLM 5.3; also the source of the AM O-lens viewer. The Qwen
ports of the same scenarios are `scheming_blackmail`, `scheming_leak` and `scheming_murder`.

|            |                                                                                                                                                                                                                                |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Status     | retired                                                                                                                                                                                                                        |
| Owner      | Kylie                                                                                                                                                                                                                          |
| Git ref    | `main` @ `7296169` (viewer only); run code is in bramble `~/workspace/agentic-misalignment` (inspect_evals 0.26 `run_all.sh` + `export.py`, no commits). Runs on HF `runs/agentic_misalignment/2026-10-24_glm53_inspect_evals` |
| Code       | [`scripts/am_olens_viewer.py`](../../../scripts/am_olens_viewer.py)                                                                                                                                                            |
| Models run | GLM 5.3                                                                                                                                                                                                                        |

## Key arms and results

- Blackmail 19/30 in the default cell.

## Transcripts and viewers

- [Docent `agentic_misalignment`](https://docent.transluce.org/dashboard/e267ec64-1a23-45d7-979e-d38fca9c48dd)
- [expview run](http://bramble:8830/#d=runs/agentic_misalignment/2026-10-24_glm53)
- [expview AM O-lens](http://bramble:8830/#d=imports/am_olens_viewer)

bramble links work on the tailnet only.
