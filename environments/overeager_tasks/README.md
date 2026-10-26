# Over-eagerness tasks × harness variants (templated grid)

Nine ambiguous or missing-value tasks (calendar, dana, gitbranch, payback, groupchat,
impossiblebench, evilgenie, rhb_bench, rhb_logs), each in three phrasings and crossed with the 8
email-ladder harness variants (base, base_home, base_done_neutral, no_done, harness, no_criterion,
plain_request, core). Does GLM act without asking? This is the templated setting × task grid.

|            |                                                                                                       |
| ---------- | ----------------------------------------------------------------------------------------------------- |
| Status     | active                                                                                                |
| Owner      | Kylie                                                                                                 |
| Git ref    | `main` @ `7296169`                                                                                    |
| Code       | [`donehack/realism/overeager/`](../../donehack/realism/overeager/)                                    |
| Run        | `uv run python realism/overeager/oe_run.py --tasks payback --variants base,core --items 16 --seeds 1` |
| Docs       | [`donehack/realism/overeager/README.md`](../../donehack/realism/overeager/README.md)                  |
| Models run | GLM 5.3, n=16–64 per cell (~6k episodes)                                                              |

## Key arms and results

- % acted without asking, base vs core: calendar **100 vs 0**, dana 81 vs 0, gitbranch 100 vs 0,
  payback 43 vs 0, groupchat 100 vs 12, impossiblebench 62 vs 0, evilgenie 56 vs 62, rhb_bench 75 vs
  37, rhb_logs 0 vs 6.
- Run dirs are only in bramble `.worktrees/realism/data/runs/overeager` (13 runs), not yet on HF.

## Transcripts and viewers

- [Docent `overeager`](https://docent.transluce.org/dashboard/495b18de-177f-4fc1-ac88-60921698706e)
- [expview overeager_v1](http://bramble:8830/#d=expview/overeager_v1)
- [expview hypotheses](http://bramble:8830/#d=expview/overeager_hyp)
- [over-eagerness summary (artifact)](https://claude.ai/code/artifact/6675ddf6-81c9-452c-8c39-2f0835156499)

bramble links work on the tailnet only.
