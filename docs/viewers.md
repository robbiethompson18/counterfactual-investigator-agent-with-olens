# Viewers: static pages on the public internet, no backend

**Status (2026-10-27):** ACTIVE. The rule for every viewer in this repo, set by Robbie on
2026-10-26; GitHub Pages enabled 2026-10-27.

## The rule

A viewer is one HTML file on Cloudflare Pages or GitHub Pages that reads the public HF dataset
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
- **GitHub Pages:** the repo has one site,
  https://robbiethompson18.github.io/counterfactual-investigator-agent-with-olens/, with Pages
  source set to GitHub Actions (enabled 2026-10-27). `.github/workflows/env-viewer-pages.yml` builds
  it from a `site/` folder, and each deploy replaces the whole site, so a second page goes into that
  same workflow's `site/` (e.g. `site/<page>/index.html`), not a second workflow.

## The one exception: Search everything (Kylie, 2026-10-27)

Kylie's call on 2026-10-27: one regex (e.g. `\bgrader`) over every read, env episode, transcript and
judge call is worth a server. The corpus is ~15 GB on HF (10 GB of batch reads, 4 GB of expview
items and token files, 1.4 GB of env episodes), too much for a browser. So
`expview/search_server.py` runs on bramble over the local mirror of the HF dataset, and the expview
page has a **Search everything** page (`#everything=1&query=…`) that calls it. It is an add-on, not
a dependency: every result links into the normal static pages (episode and token deep links that
read HF), the per-run Search page still works without it, and the button only appears where the
server answers (the copy of the page the server serves over http, or a browser whose localStorage
`expview:search_server` names it; an https page such as GitHub Pages can't call an http server, so
it stays hidden there). The server's URL stays out of docs and out of messages to anyone off the
tailnet. Details: [expview/CLAUDE.md](../expview/CLAUDE.md), "Search everything".

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

| page                                  | URL                                                                                             | source                                                                             | data on HF                          |
| ------------------------------------- | ----------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ----------------------------------- |
| CHIVE cases                           | https://chive-cases.pages.dev                                                                   | `viewer/index.html`, `scripts/build_viewer.py`, `scripts/deploy_viewer.sh`         | built-in                            |
| O-lens token viewer                   | https://olens-viewer.pages.dev                                                                  | `transcript-viewer/web/index.html`, `scripts/publish.py`, `scripts/deploy_site.sh` | `weirdchat/site/`                   |
| expview (runs, judge I/O, reads)      | https://robbiethompson18.github.io/counterfactual-investigator-agent-with-olens/expview/        | `expview/static/index.html`, `.github/workflows/env-viewer-pages.yml`              | `envs/*/*/`, `runs/**/expview/`     |
| env viewer (arms, diffs, comparisons) | https://robbiethompson18.github.io/counterfactual-investigator-agent-with-olens/ (GitHub Pages) | `environments/_lib/page.html`, `.github/workflows/env-viewer-pages.yml`            | `envs/*/spec/`, `envs/*/links.json` |

The old expview page, https://expview.pages.dev (Cloudflare), still works but is frozen at its
2026-10-26 code; expview now deploys with the env viewer to GitHub Pages under `/expview/`.

Still to decide (2026-10-26): whether the three pages share code for URL params, HF fetching and
transcript rendering, and whether any two of them are one page.

## URL parameters

Every view of a page is a link: the whole view state lives in the URL **hash** (`#key=value&…`), not
the query string. A static page has no server to read a query string, and the fragment survives the
`weirdchat-olens` → `olens-viewer` 301 and any other redirect, while a query string may not.

Rules (Robbie, 2026-10-26):

- **Keys are full words**, never single letters: `#pattern=36&transcript=…&token=134`, not
  `#p=36&t=…&tok=134`.
- **The same thing gets the same word on every page.** A token index is `token`, a free-text search
  or highlight is `query`, a dataset is `dataset`, whether the page is olens-viewer or expview.
- **Values are unchanged** by the rename (pattern indices, transcript ids, view names
  `tr|call|env|rec`).
- **Old links keep working.** Each page has one `LEGACY = {p: 'pattern', …}` map and one
  `canonHash()` that runs on load (and, for expview, on every `hashchange`): a hash with any legacy
  key is mapped to the new keys and written back with `history.replaceState`, so the address bar
  shows the canonical form with no reload and no flash. Links the page generates (copy link, bullet
  links, search result links) and the Python builders (`expview.py link`) emit only the new form.
  The hundreds of `#p=…&t=…` and `#d=…&i=…` links already in `docs/` are left as they are; the
  legacy map is what keeps them working.

| key               | olens-viewer (`transcript-viewer/web/index.html`)       | expview (`expview/static/index.html`)                            |
| ----------------- | ------------------------------------------------------- | ---------------------------------------------------------------- |
| `dataset`         | tab: `weirdchat` (default), `chive`, … (was `ds`)       | run id `runs/<experiment>/<run>`, comma list in search (was `d`) |
| `pattern`         | pattern index (was `p`)                                 |                                                                  |
| `transcript`      | transcript id within the pattern (was `t`)              |                                                                  |
| `item`            |                                                         | item id (was `i`)                                                |
| `token`           | selected token position (was `tok`)                     | token position in the transcript (was `t`)                       |
| `view`            |                                                         | `tr`, `call`, `env` or `rec` (was `v`)                           |
| `call`            |                                                         | call index, 0-based (was `c`)                                    |
| `message`         |                                                         | message index within the call (was `m`)                          |
| `file`            |                                                         | environment file path (was `f`)                                  |
| `query`           | pattern-list text filter (was `q`)                      | regex to highlight or search (was `q`)                           |
| `case`            |                                                         | `1` = case-sensitive regex                                       |
| `search`          |                                                         | `1` = the Search page (was `s`)                                  |
| `everything`      |                                                         | `1` = the Search everything page (search server)                 |
| `novel`           |                                                         | `ctx` or `transcript`: drop echo hits                            |
| `scope`           |                                                         | `text` = search transcripts and calls, not reads                 |
| `experiment`      |                                                         | search: the experiment whose runs are listed (was `env`)         |
| `reader`          |                                                         | search facet: reader / tool name (was `tool`)                    |
| `section`         |                                                         | search facet: transcript section (was `sec`)                     |
| `group`, `label`  |                                                         | search facets (`label` is `KEY=VALUE`)                           |
| `page`            |                                                         | search results page, 0-based                                     |
| `filter`          | transcript filter dropdown, e.g. `positive` (was `tf`)  |                                                                  |
| `edit`            | CHIVE edit index (was `x`)                              |                                                                  |
| `layers`          | comma list of lens layers shown, default `40` (was `L`) |                                                                  |
| `temperature`     | decode temperature of the reads shown (was `T`)         |                                                                  |
| `highlight`       | FVE highlight mode, default `joint` (was `hl`)          |                                                                  |
| `highlight_layer` | layer the highlight uses, default `40` (was `hll`)      |                                                                  |
| `overview`        | overview token offset, default `1` (was `ov`)           |                                                                  |
| `behavior`        | behavior filter on the pattern list (was `beh`)         |                                                                  |
| `sort`            | pattern-list sort                                       |                                                                  |

Only non-default values are written to the hash.
