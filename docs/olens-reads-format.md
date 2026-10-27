# Batch reads format: reads + transcripts, shown by expview with no export

**Status (2026-10-27):** ACTIVE. The one format for metamodel reads (O-lens, J++, NLA); expview
reads it in place.

A batch reader (`scripts/olens_read_requests.py`, `jpp_read_requests.py`) writes, for
`OUT = X.jsonl`:

- **`X.jsonl`**: one row per (transcript, position, layer), `id` first, plus the reader's payload
  (O-lens `bullets` + `joint_fve`, NLA `text` + `fve`, J++ `layers` + `top`). cfbench reads this.
- **`X.transcripts.jsonl`**: one line per transcript: `id`, the request's metadata (env, arm,
  positive, ...), `tokens` (token strings as Qwen saw them, so
  `tokens[row.token_index] == row.token`) and `sections` (`[[name, start, end]]` chat blocks). From
  `render.transcript_line()`; ~0.3 ms per 2k-token transcript.

`scripts/runs.py finish` runs `expview/importers/reads.py`, which writes a thin expview run at
`RUN/expview/` (or `RUN/expview/X/` for other reads files) whose items point at byte ranges of the
two files (`tokens.src`). The page fetches those ranges from HF and builds the reads in the browser
(`rowsToReads` in `expview/static/index.html`, `rows_to_reads` in `expview/expview.py`; keep them in
step). Nothing is copied. Link:
`https://robbiethompson18.github.io/counterfactual-investigator-agent-with-olens/expview/#dataset=runs/<experiment>/<run>&item=<id>`.

Gotchas:

- The offsets are into the files as uploaded: rewrite `X.jsonl` (merge, dedupe) only if you rerun
  the importer and re-upload both.
- Every Qwen and GLM-5.3 reads run of 2026-10-26/27 was backfilled with a `transcripts.jsonl`
  (scripts in commit `36f4273`: `scripts/olens_transcripts_backfill.py` for Qwen,
  `scripts/glm_olens_dense/to_reads.py` for Celeste's dense GLM parquet). Older runs have none.
- Shared-prefix reads are copied onto every transcript, so runs with long shared prompts are big
  (GLM email dense: 818 MB, 91% copies). One item fetches only its own rows; run-wide regex search
  loads the whole file.
- The NLA reader (`olens/nla` branch) should call `write_transcript` like the other two when it
  merges. olens-viewer has its own reads format and doesn't show batch reads.
