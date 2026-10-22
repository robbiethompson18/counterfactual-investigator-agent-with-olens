#!/usr/bin/env bash
# Keep at most MAX rollout containers of one env image alive: every 30 s, if fewer are running, pop the next job
# line from a shared queue file (flock'd) and launch it in the background. Also logs the box load each minute
# (timing benchmarks are load-sensitive). Each job line is
#   <results_subdir> <config.yaml> [dot.overrides ...]
# and runs one rollout (run.py --count 1) into ~/envs-results/<results_subdir>. Run from the harness dir
# (vendor/agent-interp-envs) with uv on PATH; exits when the queue is empty.
#   nohup scripts/envs_screen/queue_governor.sh <queue.txt> <image> <max> <log_dir> &
set -u
queue=$1 image=$2 max=$3 logs=$4
tick=0
while true; do
  [ $((tick % 2)) -eq 0 ] && echo "$(date -u +%FT%TZ) $(cut -d' ' -f1-3 /proc/loadavg) running=$(docker ps -q --filter ancestor="$image" | wc -l)" >> "$logs/load.log"
  tick=$((tick + 1))
  if [ "$(docker ps -q --filter ancestor="$image" | wc -l)" -lt "$max" ]; then
    job=$(flock "$queue.lock" bash -c 'line=$(head -n 1 "$0"); [ -n "$line" ] && sed -i 1d "$0"; echo "$line"' "$queue")
    [ -z "$job" ] && exit 0
    read -r subdir config overrides <<<"$job"
    echo "$(date -u +%FT%TZ) launch $job" >> "$logs/governor.log"
    # shellcheck disable=SC2086  # overrides are space-separated key=value args
    nohup uv run python scripts/run.py "$config" --network envnet --no-judge --no-build \
      --results-dir "$HOME/envs-results/$subdir" --count 1 $overrides >> "$logs/runs.log" 2>&1 &
    sleep 20  # let the container start before recounting
  fi
  sleep 30
done
