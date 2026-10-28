# Docent: our transcript viewer

**Status (2026-10-27):** ACTIVE.

Every transcript from our envs and every cfbench prediction is mirrored to public collections on
[Docent](https://docent.transluce.org) (Transluce's agent-transcript viewer). Use it to read
transcripts, filter by arm/model/flags, and link a specific transcript in docs, PRs and Slack.

## What's there

- **One collection per env, named after it** (`shutdown_palisade`, `hint_mcq`, ...): every episode
  of every `environments/` run folder, and every row of every old evalcue-harness `rows.jsonl`. Each
  Docent run's metadata has `arm`, `item`, `sample`, the run's labels (`positive` and `measure()`'s
  other values; for old rows the env's own flags), `source` (`ENV/RUN:EPISODE_ID` for run folders,
  `ENV/RUN:LINE` for old rows), `run_status` (`done`/`abandoned`), `changed_files` (workspace files
  the model changed, 5k characters each). A converted run stays in the collection its old env name
  made (hint_mcq's converted run is in `hint_cf_wrong_mcq`); new runs of the ported env go to the
  collection named after the new env.
- **`cfbench`**: every predictor transcript from `data/runs/cfbench/*/predictions.jsonl`, with the
  prediction (`p`, `k_base/n_base/k_cf/n_cf`, `condition`, `predictor`, ...) as metadata and
  `shown_links` pointing at the env transcripts the predictor was shown.
- Not there (yet): the vendored agent-interp-envs screens (their evalcue ports of sandbagging and
  funding_email are there), judge/label calls, identity experiments, WeirdChat/CHIVE. O-lens reads
  are only in the [O-lens viewer](https://olens-viewer.pages.dev).

The HF dataset (`envs/` run folders, `data/runs` for the old layout) stays the source of truth;
Docent is a derived view. Episodes with no transcript (API errors) are skipped.

## Linking

The Docent ids ship with the run on HF, so links can be looked up anywhere the run is. A run folder
has `docent_collection` in `meta.json` and `docent_run` / `docent_url` on every episode line (the
list page shows them); an old run dir has a `docent.json`
(`{"collection_id", "runs": [run id per line, or null]}`).

```bash
# one episode of a run folder, by its id (or its 0-based position in the run)
uv run --script scripts/docent_upload.py link hint_mcq/2026-10-26_or_nothink2:hint+d1~5695~61
# one old row, by the same ENV/RUN:LINE id cfbench uses (LINE is the 0-based line in rows.jsonl)
uv run --script scripts/docent_upload.py link shutdown_palisade/2026-10-22_r15_catch_words:3
# a filtered table: KEY=VALUE metadata conditions, ANDed, values parsed as JSON (true, 3, "x")
uv run --script scripts/docent_upload.py view shutdown_palisade arm=catch_bydefault_D positive=true
```

Run URLs look like `https://docent.transluce.org/dashboard/<collection>/agent_run/<run>`. To point
at one message, append `?block_idx=N` (N is the 0-based "Block N" label Docent shows on each
message; the page scrolls to it and highlights it, and fills in `transcript_id` itself). Cite the
block, not just the run, when the claim is about one tool call or read. Nested metadata is dotted
(`view ... detect_notes.bypassed=true`).

## Uploading

- Both runners upload at finish, before their HF upload, when `DOCENT_API_KEY` is set:
  `environments run` for run folders (the upload rewrites the episode shards with the ids and
  rebuilds `episodes/index.json`, so it happens before the HF step), `scripts/runs.py finish` for
  old-layout evalcue and cfbench runs. Without the key they print the retry command and only upload
  to HF. A Docent failure never blocks the HF upload.
- Catch up anything missed (pods without the key, runs from before this existed):
  `uv run --script scripts/docent_upload.py backfill` (`--dry-run` lists the candidates). It skips
  runs that are already uploaded (`docent_collection` in `meta.json`, or `docent.json`) or still
  `running`, uploads the rest, then pushes the results to HF: run folders through
  `environments upload`, old runs' new `docent.json` files in one commit. Pull other machines' runs
  from HF first or they won't be included.
- Collections are found by name and created public on first use.
- `scripts/docent_upload.py` declares its own deps (PEP 723) because the Docent SDK pulls ~90
  packages (inspect-ai, boto3, OpenTelemetry, ...). Always run it with `uv run --script`.

## Gotchas

- **Keys and ownership:** collections are administered by the Docent org "NN sprint: Robbie +
  Kylie". `DOCENT_API_KEY` should be an org service-account ("uploader") key: collections it creates
  are owned by the org, and the name lookup finds every org collection, so uploads from any machine
  land in the same collection. Kylie joins the org and mints her own uploader key. A personal key
  (Robbie's is `DOCENT_USER_API_KEY`) creates collections owned by that person, which then need
  sharing with the org by hand.
- **Run ids are random**, assigned by the SDK at upload, so a re-upload creates new runs and new
  links. Don't delete a `docent.json` to "refresh" a run; old links would break and duplicates would
  pile up.
- **Rescoring doesn't reach Docent:** a run folder's later labels files, or an old run's
  `--rescore`, don't change Docent metadata (`positive`, ...), which keeps the first upload's
  values, since re-uploading would change run ids. The labels on HF are current.
- **Docent sorts metadata alphabetically**, so column order can't be set from the upload.
- **Self-hosting Docent is not viable** (tried 2026-10-22): the open-source repo has a stale
  frontend Dockerfile, an SDK version behind PyPI, and a Next 16 bump that breaks every collection
  page.
- **Billing:** uploading and viewing use no LLM. Docent's chat/search features run on an LLM; we
  haven't added a model key, and don't need one to browse.
