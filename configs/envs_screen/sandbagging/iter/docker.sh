#!/bin/bash
# run.py --docker wrapper: swap the image run.py names (agent-interp-envs/sandbagging:base) for our overlay tag.
exec docker "${@//agent-interp-envs\/sandbagging:base/agent-interp-envs/sandbagging:iter}"
