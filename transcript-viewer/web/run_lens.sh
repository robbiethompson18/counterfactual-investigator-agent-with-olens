#!/usr/bin/env bash
# The lens read server on this machine's GPU (needs ~57 GB free), for `web/server.py --local`.
# OLENS_PYTHON: a python with torch + the olens requirements (default ../olens/.venv/bin/python).
# Log: outputs/lens.log. Stop: pkill -f 'pod/serve[r].py'
here="$(cd "$(dirname "$0")/.." && pwd)"
olens="$here/../olens"
mkdir -p "$here/outputs"
cd "$olens" && PYTHONPATH=src setsid nohup "${OLENS_PYTHON:-$olens/.venv/bin/python}" \
  "$here/pod/server.py" --idle-min 0 --max-hours 0 --no-warmup --batch 32 "$@" \
  > "$here/outputs/lens.log" 2>&1 < /dev/null &
echo "started; log: $here/outputs/lens.log"
