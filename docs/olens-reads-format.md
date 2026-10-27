# Batch reads format: reads + transcripts, shown by expview with no export

**Status (2026-10-27):** ACTIVE. The one format for metamodel reads (O-lens, J++, NLA); expview
reads it in place.

## The two files

A batch reader (`scripts/olens_read_requests.py`, `scripts/jpp_read_requests.py`) writes, for
`OUT = X.jsonl`:

- **`X.jsonl`**: one row per (transcript, position, layer), appended one transcript at a time.
  `{id, token_index, token, prefix, msg_index, role, field, char_offset, layer, ...}` plus the
  reader's payload: O-lens `bullets: [{text, fve, coef}]` + `joint_fve`; NLA `text` + `fve`; J++
  `layers` + `top` (per layer, `[token, prob]` pairs). cfbench (`cfbench/olens.py`) reads this.
- **`X.transcripts.jsonl`**: one line per transcript,
  `{id, <request metadata: env, arm, item, positive, ...>, tokens: [str], sections: [[name, start, end]]}`.
  `tokens` are the token strings exactly as Qwen saw them (`tok.decode([id])` per token), so
  `tokens[row.token_index] == row.token`. `sections` are the chat blocks (one per `<|im_start|>`, an
  assistant block split into reasoning / tool call / reply). Built by `render.transcript_line()`,
  from the render the reader already did, so it costs ~0.3 ms per 2k-token transcript against ~1–30
  s of GPU work.

Nothing else is needed to show the reads: no tokenizer, no chat template, no copy of the reads.

## How expview shows them

`scripts/runs.py finish` runs `expview/importers/reads.py RUN`. For each `X.transcripts.jsonl`
beside an `X.jsonl` it writes a thin expview run at `RUN/expview/` (for `reads.jsonl`) or
`RUN/expview/X/`: one item per transcript whose `tokens` entry is

```json
{
  "ref": "<sha1(id)[:16]>",
  "n": 147,
  "reads": 16,
  "src": {
    "transcript": ["../reads.transcripts.jsonl", 0, 1374],
    "reads": [["../reads.jsonl", 0, 23638]]
  }
}
```

i.e. byte ranges into the two files, relative to the expview run dir. The page fetches the ranges
from HF and turns the rows into reads in the browser (`rowsToReads` in `expview/static/index.html`,
`rows_to_reads` in `expview/expview.py` for `check` and `build`; keep them in step). Items carry
`extra.source` = `ENV/RUN:LINE` when the id is one (any `|suffix` dropped), so Docent links resolve.

Link:
`https://robbiethompson18.github.io/counterfactual-investigator-agent-with-olens/expview/#dataset=runs/<experiment>/<run>&item=<id>`.

## Gotchas

- The offsets are into the files as uploaded. Never rewrite `X.jsonl` (merge shards, dedupe,
  re-sort) after the expview run is built without rerunning the importer and re-uploading both.
- Reads made before 2026-10-27 have no `X.transcripts.jsonl`.
  `scripts/olens_transcripts_backfill.py RUN/X.jsonl [--requests ...]` renders the requests the same
  way (`render.render_request`), checks every row's `token`, and writes it. Done for every Qwen
  reads run of 2026-10-26/27 on HF (olens_reads cot-r3, hint-cf-v1, honest-v1, long-parity,
  hint-mcq-header, merge-v1; jpp_reads and nla_reads hint-mcq; donation_bet olens-nothink
  `reads_qwen`). `merge-v1` reads the hint-cf-v1 transcripts, so it needs
  `--requests ../2026-10-26_hint-cf-v1/requests.jsonl`.
- GLM-5.3 reads (Celeste's dense parquet) are converted by `scripts/glm_olens_dense/to_reads.py`
  into the same two files, with GLM's tokens.
- An `expview/` dir the importer didn't write (e.g. a one-off export with `tokens/` files) is left
  alone; reads files other than `reads.jsonl` go in `expview/<name>/`, so they don't collide.
- The NLA reader lives on the `olens/nla` branch; it should call `write_transcript` like the other
  two readers when it merges.
- olens-viewer (`transcript-viewer/`) still has its own reads format and doesn't show batch reads;
  use expview.
