#!/usr/bin/env bash
# Build (if the data changed) and serve the olens regex dashboard on bramble's tailnet addresses.
#   olens_search/run.sh          # serve; rebuilds the DB when any parquet / docent.json is newer than it
#   olens_search/run.sh --pull   # first re-download Celeste's readouts (picks up top-ups) and the sprint dataset's docent.json maps
# Env: OLENS_DATA (default <repo>/data/olens_search), OLENS_PORT (8821), OLENS_HOSTS (tailscale ip v4 + v6).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${OLENS_DATA:-$REPO/data/olens_search}"
RUNS_ROOT="${OLENS_RUNS_ROOT:-$REPO/data}"  # local data/runs/*/*/docent.json (see docent_aliases.json)
PORT="${OLENS_PORT:-8821}"
mkdir -p "$DATA"
if [[ "${1:-}" == "--pull" ]]; then
  [[ -f "$REPO/hf.key" ]] && export HF_TOKEN="$(cat "$REPO/hf.key")"
  uvx --from huggingface_hub hf download ceselder/glm-5.3-olens-counterfactual-investigator --repo-type dataset --local-dir "$DATA/ds"
  uvx --from huggingface_hub hf download Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens --repo-type dataset \
    --include "runs/*/*/docent.json" --local-dir "$DATA/docent"
fi
DB="$DATA/olens.duckdb"
if [[ ! -f "$DB" ]] || [[ -n "$(find "$DATA/ds" "$DATA/docent" "$RUNS_ROOT/runs" "$REPO/olens_search/docent_aliases.json" \
      \( -name '*.parquet' -o -name 'docent*.json' \) -newer "$DB" -print -quit 2>/dev/null)" ]]; then
  echo "building $DB"
  uv run --script "$REPO/olens_search/build_db.py" --ds "$DATA/ds" --docent-root "$DATA/docent" --docent-root "$RUNS_ROOT" --db "$DB"
fi
HOSTS=(${OLENS_HOSTS:-$(tailscale ip -4) $(tailscale ip -6)})
args=()
for h in "${HOSTS[@]}"; do args+=(--host "$h"); done
exec uv run --script "$REPO/olens_search/server.py" --db "$DB" --port "$PORT" "${args[@]}"
