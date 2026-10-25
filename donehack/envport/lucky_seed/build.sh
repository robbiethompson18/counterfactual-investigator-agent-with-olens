#!/usr/bin/env bash
# Build the four images: lucky_seed / lucky_seed_readable_grader, agent uid 1000 / 200.
set -euo pipefail
cd "$(dirname "$0")"
AIE=${AIE_DIR:-$HOME/workspace/agent-interp-envs}  # a checkout of gkroiz/agent-interp-envs
CTX=(--build-context "aie=$AIE/environments/lucky_seed_readable_grader")
for uid in 1000 200; do
  sfx=$([ $uid = 200 ] && echo -u200 || true)
  podman build "${CTX[@]}" --build-arg AGENT_UID=$uid -t localhost/envport-lucky_seed$sfx .
  podman build "${CTX[@]}" --build-arg AGENT_UID=$uid --build-arg GRADING_LEAK=/srv/grading-runner \
    -t localhost/envport-lucky_seed_readable_grader$sfx .
done
