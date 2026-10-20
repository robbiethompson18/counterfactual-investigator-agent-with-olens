#!/usr/bin/env bash
# Start the viewer detached (log: outputs/web.log). Binds to $WCO_BIND (default 127.0.0.1), port 8803.
# Add --local to read on this machine's GPU (web/run_lens.sh) instead of a cloud pod.
# Stop: pkill -f '^.venv/bin/python web/serve[r].py'
cd "$(dirname "$0")/.." && mkdir -p outputs
setsid nohup .venv/bin/python web/server.py "$@" > outputs/web.log 2>&1 < /dev/null &
echo "started; log: $PWD/outputs/web.log"
