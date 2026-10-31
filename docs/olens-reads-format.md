# Batch reads format: reads + transcripts, shown by expview with no export

**Status (2026-10-31):** ACTIVE. The one format for metamodel reads (O-lens, J++, NLA); expview
reads it in place.

A batch reader (`scripts/olens_read_requests.py`, `jpp_read_requests.py`) writes, for
`OUT = X.jsonl`:

- **`X.jsonl`**: one row per (transcript, position, layer), `id` first, plus the reader's payload
  (O-lens `bullets` + `joint_fve`, NLA `text` + `fve`, J++ `layers` + `top`). cfbench reads this.
- **`X.transcripts.jsonl`**: one line per transcript: `id`, the request's metadata (env, arm,
  positive, ...), `tokens` (token strings as Qwen saw them, so
  `tokens[row.token_index] == row.token`) and `sections` (`[[name, start, end]]` chat blocks). From
  `render.transcript_line()`; ~0.3 ms per 2k-token transcript.

Each read row places its token with `msg_index` (template tokens get the message before them, -1
before the first), `field` (`content`, `reasoning_content`, `tool_call`, or null for template
tokens) and `char_offset`: the token's first character in `messages[msg_index][field]` as given,
leading whitespace counted (the chat template trims it), or in that tool-call argument's string
value. Request rows' `char_positions` use the same offsets. `render.render_request` finds each field
by rendering the messages again with a sentinel in place of each field's text. Reads made before
2026-10-31 (PR PRNUM) differ in two ways, in the saved runs only on transcripts with an assistant or
tool turn: `char_offset` counted from the stripped field, so on a field with leading whitespace it
is that many chars early (968 of the 113,889 saved `messages` request rows); and fields were matched
with `str.find` from the previous field's end, so a short field could match template text (an answer
`e` inside `<|im_end|>`, a `bash` argument inside `<function=bash>`), mislabelling the tokens after
it (37 rows, 4 distinct transcripts). Single-turn prompts, `text` rows and every read position were
unaffected.

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

## Known issues

- **Split characters in saved `tokens`.** `X.transcripts.jsonl` `tokens` (and each row's `token`)
  are per-token decodes, so a character split across tokens (an emoji, some CJK) shows as U+FFFD in
  each piece. Re-render the request (`render.render_request`) for the exact text; the reads
  themselves are unaffected.
- **J++ provenance has no lens commit.** `jpp_read_requests.py` writes `lens.revision` from the
  resolved HF cache path (`olens/src/olens/jpp.py`), which is the blob's directory: every J++ run so
  far says `"revision": "blobs"`, not the `neuronpedia/jacobian-lens` commit.
