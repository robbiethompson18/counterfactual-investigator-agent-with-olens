# Evolve tier-1 node m1023190842-4480

donehack_reply_end: the same is_prime mechanism from another lineage. One of four tier-1 nodes
picked by Claude from 39 shortlisted nodes of evolve run v1 (209 nodes), each a baseline env + a
REAL edit that lowers the rate + a NULL edit that looks like it should but doesn't.

|            |                                                                                                                                                                     |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Status     | active                                                                                                                                                              |
| Owner      | Kylie                                                                                                                                                               |
| Git ref    | `main` @ `7296169` (evolve loop, PRs #114/#119); node files (baseline/, real/, null/, grader.py, hypothesis.md) on HF `runs/evolve/data/v1/nodes/m1023190842-4480/` |
| Code       | [`evolve/`](../../evolve/)                                                                                                                                          |
| Run        | `evolve status v1   # see evolve/README.md`                                                                                                                         |
| Docs       | [`evolve/README.md`](../../evolve/README.md)                                                                                                                        |
| Models run | Qwen3.6-27B (shared vLLM server)                                                                                                                                    |

## Key arms and results

- REAL: the subtlest edit, rename `reward` → `all_passed` (removes the `reward = 1.0` computation).
  14/30 → 7/30.
- NULL: drop "will terminate the harness" (a strong decoy; forecasters at chance). 13/30. Needs more
  samples.

## Transcripts and viewers

- [Docent `evolve_tier1`](https://docent.transluce.org/dashboard/25527074-a27b-48a2-9231-60d753be0dcf)
- [evolve dashboard (tailnet)](http://bramble:8815/)

bramble links work on the tailnet only.
