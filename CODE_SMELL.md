# Code smells

Tech debt we know about, newest first.

## 2026-10-31: expview exports over HF's 10,000-files-per-directory limit

`environments evals NAME --upload` fails when a cfbench run's expview export writes more than 10k
files to `expview/tokens/` (one per prediction with reads; `2026-10-30_essay-cf-haiku-olens` has
10,800, so its tokens were left off HF). Shard `tokens/` into subdirectories (e.g. by id prefix) in
`environments/_lib/evals.py` `export_expview` / expview's `Run`.
