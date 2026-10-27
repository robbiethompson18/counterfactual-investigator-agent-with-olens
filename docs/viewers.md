# Viewers: static pages on the public internet, no backend

**Status (2026-10-26):** ACTIVE. The rule for every viewer in this repo, set by Robbie on
2026-10-26.

## The rule

A viewer is one HTML file on Cloudflare Pages that reads the public HF dataset
([Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens))
directly. No server process, nothing on bramble or a tailnet, no SkyPilot pod behind a button.

- **Data:** whatever the page needs is published by the job that made it (`scripts/runs.py finish`,
  `transcript-viewer/scripts/publish.py`). The page fetches it with plain `fetch()` from
  `https://huggingface.co/datasets/<repo>/resolve/main/<path>`. HF sends CORS headers for any origin
  and supports range requests. The HF tree API (`/api/datasets/<repo>/tree/main/<dir>`) lists files.
- **Search:** pull the whole run (or episode) and run a JavaScript `RegExp` over it in the browser.
  A run's reads are a few MB gzipped; an episode's dense reads are a few hundred KB. Cross-run
  search is "load the runs for this env", lazily. No DuckDB, no DuckDB-WASM, no server-side index. A
  regex over millions of reads at once is a one-off `grep` or Python script over the parquet, not a
  web feature.
- **GPU:** reads are batch jobs (`scripts/olens_read_requests.py` on RunPod,
  `runpod/olens_reads.sh`) whose outputs land on HF. A viewer shows what is there. It never starts a
  pod or queues a read.
- **Links:** every link in a doc, PR, Slack message or the report must open for Neel and anyone else
  with the URL. `bramble:*` and other tailnet URLs are not links.
- **Deploy:** `wrangler pages deploy` with `CLOUDFLARE_API_TOKEN` from `.envrc.local`; see
  `scripts/deploy_viewer.sh`. Deploy a preview branch and check two existing deep links before
  deploying `main`.

## Why

As of 2026-10-26 the repo had six viewers: `transcript-viewer/web` (a 969-line Python server plus a
937-line page), `expview` (FastAPI + a DuckDB regex index + an HF pull timer on bramble),
`olens_search` (FastAPI + DuckDB on bramble), `baseline_hyp/browser`, `weirdness_elo/server.py` and
`donehack/viewer`, next to the two static ones that worked: `viewer/` (chive-cases.pages.dev) and
olens-viewer.pages.dev. The backends existed for two reasons, and neither held up:

- **Cross-run regex search.** Zero docs link an expview search; the only consumer of `olens_search`
  was one null study. In practice we grep.
- **Live GPU reads from the page.** Every read cited in a doc came from the batch reader. The "Start
  GPU" button, read queue and SkyPilot fleet were one person's local workflow.

Meanwhile the tailnet-only URLs could not go in the report or a message to anyone outside the two of
us, which is the whole point of a viewer.

Checked and rejected the same day:

- **DuckDB-WASM over parquet on HF:** a 5–10 MB bundle and seconds per regex over 100 MB of parquet
  unless you do row-group pushdown. Ugly and slow.
- **HF datasets-server `/search` and `/filter`:** hosted, CORS-enabled, no backend. `/search` is
  case-insensitive substring over all string columns against an index HF builds on first touch;
  `/filter` takes a SQL `WHERE` with `LIKE` but rejects `regexp_matches`. On
  `ceselder/glm-5.3-olens-counterfactual-investigator` a 450-row config answered in 4 s cold and 0.2
  s for a filter; a 14k-row config and the 228k-row dense config were still "index is loading" after
  six minutes, and the API returned empty bodies under repeated calls. Fine for a few thousand rows,
  not something to build on. It also needs a real HF dataset (configs, parquet or jsonl), and our
  sprint dataset is a file dump (`is-valid` is false on every capability).

## Where things are

| page                             | URL                            | source                                                                             | data on HF         |
| -------------------------------- | ------------------------------ | ---------------------------------------------------------------------------------- | ------------------ |
| CHIVE cases                      | https://chive-cases.pages.dev  | `viewer/index.html`, `scripts/build_viewer.py`, `scripts/deploy_viewer.sh`         | built-in           |
| O-lens token viewer              | https://olens-viewer.pages.dev | `transcript-viewer/web/index.html`, `scripts/publish.py`, `scripts/deploy_site.sh` | `weirdchat/site/`  |
| expview (runs, judge I/O, reads) | https://expview.pages.dev      | `expview/static/index.html`, `expview/deploy.sh`                                   | `runs/**/expview/` |

Still to decide (2026-10-26): whether the three pages share code for URL params, HF fetching and
transcript rendering, and whether any two of them are one page.
