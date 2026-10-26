# AISI alignment-honeypot replication

Replicates UK AISI's "open-weight model that games alignment honeypots" on GLM 5.3: resample fixed
scenarios and compare misbehaviour in samples that verbalise eval awareness (VEA) vs not, on Agentic
Misalignment (AM), Misalignment Continuation (MC) and AbstentionBench.

|            |                                                                                                                                                                                                                                      |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Status     | retired                                                                                                                                                                                                                              |
| Owner      | Kylie                                                                                                                                                                                                                                |
| Git ref    | not in this repo: bramble `~/workspace/glm-evalgaming` (local git repo, no remote, HEAD `8272fda`: run*am_fixed.sh, run_mc.sh, run_abst*\*.sh, monitor.py, export.py). Runs on HF `runs/eval_gaming/2026-10-24_glm53_aisi_honeypots` |
| Models run | GLM 5.3; awareness monitor ours (Sonnet 5.5 → 4.5 → GPT-5.6 fallback), MC grader Sonnet 4.5                                                                                                                                          |

## Key arms and results

- AM fixed blackmail scenario: aware 74%; **blackmail 16% when aware vs 51% unaware** (29/185 vs
  33/65, p=8e-8).
- AM full set (450): 5% vs 10% (p=0.06). MC drone: 24% vs 76%, but awareness often follows the
  decision to stop.
- AbstentionBench: null (41% vs 41%).

## Transcripts and viewers

- [Docent `aisi_eval_gaming`](https://docent.transluce.org/dashboard/28a181a0-b9a2-40bb-a6b5-de1ae849a899)
- [judge I/O (artifact)](https://claude.ai/artifact/EUhCnWS5MJ5EreQDF22U9B)
- [expview glm_evalgaming_all](http://bramble:8830/#d=expview/glm_evalgaming_all)

bramble links work on the tailnet only.
