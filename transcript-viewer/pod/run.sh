#!/usr/bin/env bash
# The read server, run as the sky job "server" (see web/pod.py). Exits on idle; the pod then autodowns.
unset CUDA_VISIBLE_DEVICES   # sky exec without --gpus hides the GPU from the job
cd ~/olens && PYTHONPATH=src HF_HOME=/data/hf TRITON_CACHE_AUTOTUNING=1 \
  exec .venv/bin/python ~/sky_workdir/pod/server.py "$@"
