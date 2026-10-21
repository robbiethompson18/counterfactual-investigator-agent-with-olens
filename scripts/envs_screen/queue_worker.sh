#!/usr/bin/env bash
# One rollout slot: pop the next job line from a shared queue file (flock'd) and run it, until the queue is empty.
# Start K of these to keep K containers busy; append lines to the queue to add work. Each job line is
#   <results_subdir> <config.yaml> [dot.overrides ...]
# and runs one rollout (run.py --count 1) into ~/envs-results/<results_subdir>. Run from the harness dir
# (vendor/agent-interp-envs) with uv on PATH.
#   nohup scripts/envs_screen/queue_worker.sh ~/envs-results/optimize_cache_c/queue.txt >> worker.log 2>&1 &
set -u
queue=$1
while true; do
  job=$(flock "$queue.lock" bash -c 'line=$(head -n 1 "$0"); [ -n "$line" ] && sed -i 1d "$0"; echo "$line"' "$queue")
  [ -z "$job" ] && exit 0
  read -r subdir config overrides <<<"$job"
  echo "$(date -u +%FT%TZ) start $job"
  # shellcheck disable=SC2086  # overrides are space-separated key=value args
  uv run python scripts/run.py "$config" --network envnet --no-judge --no-build \
    --results-dir "$HOME/envs-results/$subdir" --count 1 $overrides
  echo "$(date -u +%FT%TZ) end rc=$? $job"
done
