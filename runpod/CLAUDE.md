# runpod/: Robbie's GPU pods

This directory is Robbie's code, for agents working for Robbie. Robbie and Kylie keep their GPU
compute separate:

- **Robbie** (and his agents) use plain RunPod pods on his account, via `runpodctl` and
  `runpod/pod.sh`. Never SkyPilot (`sky`, `transcript-viewer/sky/`), never the bramble GPU, and never
  anything that launches GPUs from bramble (e.g. the bramble:8801 lens explorer's Start GPU button
  or its local backend). Those are Kylie's accounts and hardware. Reading her viewers and caches is
  fine.
- **Kylie** (and her agents) use SkyPilot and bramble. If you're working for Kylie, this directory
  isn't for you.

## pod.sh

```
runpod/pod.sh up NAME [GPU_ID] [GPU_COUNT]   # create, wait for ssh; default 1x "NVIDIA H200"
runpod/pod.sh ssh NAME [CMD...]
runpod/pod.sh sync NAME                      # rsync the repo to /workspace/repo
runpod/pod.sh down NAME                      # delete
runpod/pod.sh list
```

Needs `RUNPOD_API_KEY` (in `.envrc.local`, loaded by direnv) and a logged-in `hf` CLI. GPU ids come
from `runpodctl gpu list`. An H200 is about $5/hr. Delete pods the moment a job finishes.

Gotchas:

- `runpodctl pod create --env` only reaches PID 1, not ssh sessions. `up` copies the `HF_*` vars
  into `/etc/environment` so `ssh` commands see `HF_TOKEN`.
- `sync` skips `.envrc.local`, `.venv`, `data/` and `checkpoints/`, so secrets and big files stay
  local. Don't copy `.envrc.local` to a pod.
- For O-lens reads on a pod, follow `olens/README.md` (deps, `scripts/download.sh`,
  `PYTHONPATH=src`). Qwen3.6-27B in bf16 is ~54 GB, so it needs an 80 GB+ GPU.
