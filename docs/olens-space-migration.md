# Old-lens reads to redo with the space O-lens

Until 2026-10-22 ~17:25 PDT every O-lens read we made used agu18dec `olens_s3d_rl600`, which opens
~half its bullets with a Chinese word (CLAUDE.md). The space lens `olens_s3d_space_rl600` replaces
it. This is the list of what still holds old-lens reads, to re-read once the new-read queue
(`runpod/olens_reads.sh`) is empty. Tick items off here as they're redone.

## Redone already

- `data/runs/olens_reads/2026-10-22_{cfbench-spike-v1,overnight-v1,overnight-v2-targets}`: read
  again from scratch with the space lens; the first 57 old-lens transcripts are kept as
  `reads_old_lens_rl600.jsonl` in the spike run.

## To redo

Inventory as of 2026-10-22. "Easy" means the token ids and read positions are on disk, so the batch
reader can redo them once it takes `token_ids` + `positions` rows (not yet: it reads every stop
token of a rendered chat). Then republish/re-sync whatever shows them.

| Artifact                                                                                                                              | Rows                                         | Re-read                                                                              |
| ------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------ |
| `transcript-viewer/data/reads.jsonl` (viewer cache: WeirdChat favourites, confession follow-ups, thinking resamples)                  | 49,799 reads, 34 transcripts, L24-56 step 8  | easy                                                                                 |
| `transcript-viewer/data/weirdchat/reads/{bramble,hospital-soft}.jsonl` (other machines' viewer reads, on HF)                          | 283,835 + 11,550, 294 transcripts, 11 layers | easy, but big                                                                        |
| `data/runs/organisms/olens_reads.jsonl`                                                                                               | 19,170 reads, 82 rollouts x 2 conds          | easy (organism capture needs its LoRA)                                               |
| `data/runs/stoptokens/reads_*.jsonl` (+ `view_*.txt`)                                                                                 | 23,247                                       | easy                                                                                 |
| `data/runs/vardania_sneeze/olens_reads.jsonl`                                                                                         | 11,382                                       | easy                                                                                 |
| `data/runs/party_resample/paired.jsonl` lens fields                                                                                   | 2,068                                        | easy, then re-pair                                                                   |
| `data/runs/identity/2026-10-21_*/lens*.jsonl` (+ `combo_google/grade_lens/` labels)                                                   | ~9k                                          | medium: rebuild prompts from `probes.json`, re-grade                                 |
| `data/runs/cfbench/2026-10-22_gflash-olens-*` (predictions that consumed old spike reads)                                             | 10 runs                                      | re-run the predictors on the new spike reads                                         |
| Public: olens-viewer.pages.dev, HF `weirdchat/reads/*` and `runs/*`, `orbita2d/investigator-agent` parquet, Docent cfbench runs above |                                              | republish / re-sync / mark superseded (don't delete `docent.json`, `docs/docent.md`) |

Not ours, leave: Agam's banks (`data/banks/agam/`, `data/banks/olens_rl/`, his harness's
`s3d_rl600_all`), chive-cases.pages.dev. Keep: `data/runs/olens_space_fix/` (old vs new on purpose).

Code: fixed on 2026-10-22 (`AguLens` default, `bullet_prefix`, the batch reader's merged path, the
viewer pod server). A viewer pod that was already running keeps the old lens until restarted.
