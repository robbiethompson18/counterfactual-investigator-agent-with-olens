# Old-lens reads to redo with the space O-lens

**Status (2026-10-26):** MIXED. "Redone already" is accurate. The "To redo" table has had no
progress since 2026-10-23 and should be read as abandoned: no viewer cache has a space-lens row, the
batch reader still can't take `token_ids` + `positions`, the 372 unread transcripts were never
requeued. The gflash row was closed by excluding old-lens runs from scoring (`--allow-old-lens`),
not by re-running.

Until 2026-10-22 ~17:25 PDT every O-lens read we made used agu18dec `olens_s3d_rl600`, which opens
~half its bullets with a Chinese word (CLAUDE.md). The space lens `olens_s3d_space_rl600` replaces
it. This is the list of what still holds old-lens reads, to re-read once the new-read queue
(`runpod/olens_reads.sh`) is empty. Tick items off here as they're redone.

## Redone already

- `data/runs/olens_reads/2026-10-22_{cfbench-spike-v1,overnight-v1,overnight-v2-targets}`: read
  again from scratch with the space lens; the first 58 old-lens transcripts (71,310 rows, all spike
  transcripts) are kept as `reads_old_lens_rl600.jsonl` in the spike run.
- Metadata (2026-10-23): every cfbench run that consumed reads has `params.olens_lens`. The
  `gflash-olens-*` runs ran on old-lens reads but pointed at the spike's `reads.jsonl`, which was
  later rewritten with the space lens; they now point at `reads_old_lens_rl600.jsonl` (`meta_fixes`
  says why). Old-lens dirs without a `meta.json` (`runs/organisms`, `runs/party_resample`,
  `weirdchat/reads`, `runs/stoptokens/2026-10-20_stop-token-reads` and
  `runs/vardania_sneeze/2026-10-20_great-sneeze`, backfilled to HF 2026-10-31) have a `LENS.md`.
  New cfbench runs record `params.olens_sources` (lens and row count per reads file) themselves.
- Viewer caches (`transcript-viewer/data/reads.jsonl`, `weirdchat/reads/*`): every row predates the
  switch. Since 2026-10-23 the viewer tags each read with its `lens` (rows without one are
  `olens_s3d_rl600`) and puts the lens in the key, so the space lens re-reads instead of hitting the
  old cache. The page shows an old read, marked "old lens", until the space lens has read that
  token.

## Unread when the budget ran out (2026-10-22 ~22:25 PDT)

372 queued transcripts were never read: 258 cfbench-spike P2 (3 layers each), most of overnight-v1
and 5 of overnight-v2-targets, plus 5 spike P1. They're in
`data/olens_requests/unread_2026-10-22.jsonl` (priorities and per-row `layers` kept); re-queue that
file with `runpod/olens_reads.sh` into the same run dirs once there is GPU budget. The runs are on
HF with status `abandoned`.

## To redo

**Abandoned (2026-10-26).** Nothing in this table was redone after 2026-10-23; it stays as the
inventory of which artifacts hold old-lens reads.

Inventory as of 2026-10-22. "Easy" means the token ids and read positions are on disk, so the batch
reader can redo them once it takes `token_ids` + `positions` rows (not yet: it reads every stop
token of a rendered chat). Then republish/re-sync whatever shows them.

| Artifact                                                                                                                              | Rows                                         | Re-read                                                                              |
| ------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------ |
| `transcript-viewer/data/reads.jsonl` (viewer cache: WeirdChat favourites, confession follow-ups, thinking resamples)                  | 49,799 reads, 34 transcripts, L24-56 step 8  | easy                                                                                 |
| `transcript-viewer/data/weirdchat/reads/{bramble,hospital-soft}.jsonl` (other machines' viewer reads, on HF)                          | 283,835 + 11,550, 294 transcripts, 11 layers | easy, but big                                                                        |
| `data/runs/organisms/olens_reads.jsonl`                                                                                               | 19,170 reads, 82 rollouts x 2 conds          | easy (organism capture needs its LoRA)                                               |
| `data/runs/stoptokens/2026-10-20_stop-token-reads/reads_*.jsonl` (+ `view_*.txt`)                                                    | 23,247                                       | easy                                                                                 |
| `data/runs/vardania_sneeze/2026-10-20_great-sneeze/olens_reads.jsonl`                                                                 | 11,382                                       | easy                                                                                 |
| `data/runs/party_resample/paired.jsonl` lens fields                                                                                   | 2,068                                        | easy, then re-pair                                                                   |
| `data/runs/identity/2026-10-21_*/lens*.jsonl` (+ `combo_google/grade_lens/` labels)                                                   | ~9k                                          | medium: rebuild prompts from `probes.json`, re-grade                                 |
| `data/runs/cfbench/2026-10-22_gflash-olens-*` (predictions that consumed old spike reads)                                             | 10 runs                                      | re-run the predictors on the new spike reads                                         |
| Public: olens-viewer.pages.dev, HF `weirdchat/reads/*` and `runs/*`, `orbita2d/investigator-agent` parquet, Docent cfbench runs above |                                              | republish / re-sync / mark superseded (don't delete `docent.json`, `docs/docent.md`) |

Not ours, leave: Agam's banks (`data/banks/agam/`, `data/banks/olens_rl/`, his harness's
`s3d_rl600_all`), chive-cases.pages.dev. Keep: `data/runs/olens_space_fix/` (old vs new on purpose).

Code: fixed on 2026-10-22 (`AguLens` default, `bullet_prefix`, the batch reader's merged path, the
viewer pod server). A viewer pod that was already running keeps the old lens until restarted.
