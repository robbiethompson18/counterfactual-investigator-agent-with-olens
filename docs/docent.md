# Docent: our transcript viewer

Every transcript from our envs and every cfbench prediction is mirrored to public collections on
[Docent](https://docent.transluce.org) (Transluce's agent-transcript viewer). Use it to read
transcripts, filter by arm/model/flags, and link a specific transcript in docs, PRs and Slack.

## What's there

- **One collection per env** (`shutdown_palisade`, `hiring_resume_names_borderline`, ...): every row
  of every evalcue-harness `rows.jsonl`. Each run's metadata has `arm`, `model`, `item`, `sample`,
  the env's own flags (`positive`, `bypassed`, `decision`, ...), `source` (`ENV/RUN:LINE`),
  `run_status` (`done`/`abandoned`), `changed_files` (workspace files the model changed, 5k
  characters each).
- **`cfbench`**: every predictor transcript from `data/runs/cfbench/*/predictions.jsonl`, with the
  prediction (`p`, `k_base/n_base/k_cf/n_cf`, `condition`, `predictor`, ...) as metadata and
  `shown_links` pointing at the env transcripts the predictor was shown.
- Not there (yet): the vendored agent-interp-envs screens (their evalcue ports of sandbagging and
  funding_email are there), judge/label calls, identity experiments, WeirdChat/CHIVE. O-lens reads
  are only in the [O-lens viewer](https://olens-viewer.pages.dev).

`data/runs` on HF stays the source of truth; Docent is a derived view. Rows with no transcript (API
errors) are skipped.

## Linking

Each uploaded run dir gets a `docent.json` (`{"collection_id", "runs": [run id per line, or null]}`)
that syncs to HF with the run, so links can be looked up anywhere the run is:

```bash
# one transcript, by the same ENV/RUN:LINE id cfbench uses (LINE is the 0-based line in rows.jsonl)
uv run --script scripts/docent_upload.py link shutdown_palisade/2026-10-22_r15_catch_words:3
# a filtered table: KEY=VALUE metadata conditions, ANDed, values parsed as JSON (true, 3, "x")
uv run --script scripts/docent_upload.py view shutdown_palisade arm=catch_bydefault_D positive=true
```

Run URLs look like `https://docent.transluce.org/dashboard/<collection>/agent_run/<run>`. Nested
metadata is dotted (`view ... detect_notes.bypassed=true`).

## Uploading

- `scripts/runs.py finish` uploads the run (before its HF sync) when `DOCENT_API_KEY` is set, so new
  evalcue and cfbench runs appear automatically. Without the key it prints a note and only syncs to
  HF. A Docent failure never blocks the HF sync.
- Catch up anything missed (pods without the key, runs from before this existed):
  `uv run --script scripts/docent_upload.py backfill`. It skips runs that have `docent.json` or are
  still `running`, uploads the rest, then pushes all new `docent.json` files to HF in one commit.
  Pull other machines' runs from HF first or they won't be included.
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
- **Rescoring doesn't reach Docent:** after `evalcue.run --rescore`, Docent metadata (`positive`,
  ...) keeps the first upload's values, since re-uploading would change run ids. `rows.jsonl` on HF
  is current.
- **Docent sorts metadata alphabetically**, so column order can't be set from the upload.
- **Self-hosting Docent is not viable** (tried 2026-10-22): the open-source repo has a stale
  frontend Dockerfile, an SDK version behind PyPI, and a Next 16 bump that breaks every collection
  page.
- **Billing:** uploading and viewing use no LLM. Docent's chat/search features run on an LLM; we
  haven't added a model key, and don't need one to browse.
