#!/bin/bash
# On robbie-box: launch arms of one stage in parallel through the shared harness, our overlay image and the gateway.
#   configs/sandbagging_iter/launch.sh <stage> <count per arm> <concurrency per arm> <arm>...
# Results: ~/envs-results/sandbagging-iter/<stage>/<arm>/sandbagging/Qwen-Qwen3.6-27B/<timestamp>/run-N
set -e
stage=$1 count=$2 conc=$3; shift 3
cfgs=$PWD/configs/sandbagging_iter
cd ~/repos/cia-envs/vendor/agent-interp-envs
export PATH=$HOME/.local/bin:$PATH
for arm in "$@"; do
  uv run python scripts/run.py "$cfgs/$arm.yaml" --docker "$cfgs/docker.sh" --no-build --network envnet --no-judge \
    --results-dir ~/envs-results/sandbagging-iter/$stage/$arm --count "$count" --max-concurrent "$conc" \
    agent.provider=fireworks agent.model=Qwen/Qwen3.6-27B agent.reasoning_effort=high \
    > ~/envs-results/sandbagging-iter/$stage.$arm.log 2>&1 &
done
wait
