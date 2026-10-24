# olens_search: regex dashboard over the GLM-5.3 O-lens readouts

A small web app for regex search over every reader output in
[ceselder/glm-5.3-olens-counterfactual-investigator](https://huggingface.co/datasets/ceselder/glm-5.3-olens-counterfactual-investigator)
(Celeste's readouts of our GLM-5.3 transcripts): GLM olens SFT and RL (greedy + 4 samples), the Qwen3.6-27B distilled and
from-scratch readers, and the J-lens L60 / L51 top-10 tokens. About 4.8M reader texts over 356k read positions.

- Search: RE2 regex (case-insensitive by default), filters for config / env / arm / reader / position kind / role / label.
  "Novel" drops a hit when the regex also matches the position's ±40-token context and true next 16 tokens, or (stricter)
  anywhere in the rendered transcript. Results show the context tail, the read token, next 16 and the matching reads, plus
  a per (config, env, arm) table: episodes with a hit and the positive / aware rate among them vs the whole slice.
  CSV export of all matching reads.
- Episode view: the full rendered transcript (system / user / reasoning / assistant / tool blocks) with a marker at every
  read position (yellow when one of its reads matches the regex). Clicking a marker shows every reader's output there.
  Search hits link straight to their position. Each episode links to its Docent run when one exists.

## Run

```bash
olens_search/run.sh          # builds data/olens_search/olens.duckdb if inputs changed, then serves on the tailnet
olens_search/run.sh --pull   # first re-download Celeste's dataset (new top-ups) and the sprint dataset's docent.json maps
```

Serves on `http://bramble:8821/`, bound to bramble's Tailscale v4 + v6 addresses only (`OLENS_HOSTS`, `OLENS_PORT` to
change). Data lives in `data/olens_search/` (`ds/` = the HF download, `docent/` = docent.json maps); `OLENS_DATA` and
`OLENS_RUNS_ROOT` override the locations. Building the DB takes ~15 s.

To keep it running across sessions, install the systemd user unit (see the header of `olens-search.service`), then
`systemctl --user restart olens-search` after new data lands.

## Files

- `build_db.py`: parquet -> DuckDB (`positions`, `reads` long table, `episodes` with rendered text + Docent URL). Locates
  each read token's character offset in the rendered text from its left context.
- `docent_aliases.json`: Celeste's run dirs use the HF names; Docent maps were uploaded from the local run-dir names. This
  maps one to the other (honeypots per source file; row order is identical).
- `server.py`: FastAPI + DuckDB, one listening socket per `--host`.
- `static/index.html`: the UI (no build step).
