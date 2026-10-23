# What lives where in our HF dataset

Our data is the public HF dataset
[Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens),
mirrored by the gitignored `data/` (`hf download ... --repo-type dataset --local-dir data`). Its
dataset card (`data/README.md`) is older than most of this.

## Top level

| Path         | What                                                                                        | Written by                                              |
| ------------ | ------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| `banks/`     | `olens_rl/<prompt_id>.json.gz`: Agam's CHIVE O-lens banks (old `olens_s3d_rl600`), not ours | `scripts/fetch_olens_banks.py`                          |
| `runs/`      | every experiment run, `runs/<experiment>/<YYYY-MM-DD>_<slug>/` with a `meta.json`           | `scripts/runs.py finish` / `sync`                       |
| `weirdchat/` | the transcript viewer's shared data: reads, rollouts, follow-ups, stance, the static site   | `transcript-viewer/scripts/publish.py` (see its README) |
| `viewer/`    | the CHIVE run viewer's comparison files                                                     | `scripts/build_viewer.py`                               |

`uv run python scripts/runs.py ls` prints one line per local run (experiment, status, description).
A run's `meta.json` has the git SHA, command, models with revisions, where they were served, and
params. Env runs (`runs/<env>/`) are documented in their env docs (`docs/envs/`, `evalcue/`,
`cfbench/`).

## O-lens reads

Which lens produced a read matters: until 2026-10-22 ~17:25 PDT everything used the old
`olens_s3d_rl600` (Chinese-prefixed bullets, AR scored bullets with no leading space), and after it
the space lens `olens_s3d_space_rl600` (AR scores `" " + bullet`). Every run below records it in
`meta.json` `params.lens` and `params.ar_bullet_prefix`; runs from `scripts/olens_read_requests.py`
after 2026-10-23 also carry `reads.provenance.json` (`olens.provenance`). The redo list for old-lens
reads is [olens-space-migration.md](olens-space-migration.md).

| Run dir (`runs/...`)                                | Lens  | Contents                                                                                                                                                                                                                            |
| --------------------------------------------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `olens_reads/<date>_<slug>/`                        | space | `requests.jsonl` (transcripts), `reads.jsonl` (one row per stop token × layer: bullets, per-bullet + joint FVE; format in `scripts/olens_read_requests.py`), sometimes `reads_old_lens_rl600.jsonl`                                 |
| `olens_residual_reads/2026-10-23_v1/`               | space | (synced when the run finishes) reads of h with the realized continuation projected out (`scripts/olens_residual_reads.py`); `atoms/<id, "/" → "__">.pt`: whitened z, 32 continuation-prefix atoms, plain-read bullet atoms, L44/L56 |
| `olens_novelty/<date>_<slug>/`                      | space | atoms for an `olens_reads` run (`scripts/olens_atoms.py`, 6 prefix lengths, one layer, text included) and `novelty.jsonl` + `novelty_extremes.md` (`scripts/olens_novelty.py`)                                                      |
| `olens_space_fix/2026-10-22_favourites-old-vs-new/` | both  | the 3 favourite WeirdChat transcripts read by old and space lens, each AR-scored with and without the space (on purpose)                                                                                                            |
| `jpp_reads/`, `nla_reads/`                          | n/a   | J++ lens and NLA reads of the cfbench spike set                                                                                                                                                                                     |
| `identity/2026-10-21_*`                             | old   | identity-belief probes with O-lens reads                                                                                                                                                                                            |
| `organisms/`, `party_resample/`                     | old   | pre-`runs.py` dirs with no `meta.json`; files at the experiment level. `data/runs/stoptokens/` and `vardania_sneeze/` (also old lens) were never synced and exist only in Robbie's local `data/`                                    |

`weirdchat/reads/*.jsonl` (the viewer's per-machine read caches) are old-lens too until re-read.
