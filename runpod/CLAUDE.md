# runpod/: Robbie's GPU pods

This directory is Robbie's code, for agents working for Robbie. Robbie and Kylie keep their GPU
compute separate:

- **Robbie** (and his agents) use plain RunPod pods on his account, via `runpodctl` and
  `runpod/pod.sh`. Never SkyPilot (`sky`, `transcript-viewer/sky/`), never the bramble GPU, and
  never anything that launches GPUs from bramble (e.g. the bramble:8801 lens explorer's Start GPU
  button or its local backend). Those are Kylie's accounts and hardware. Reading her viewers and
  caches is fine.
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
- The image sets `HF_HOME=/workspace/.cache/huggingface` (and `up` copies it into
  `/etc/environment`). Don't set a different `HF_HOME` on the pod: it downloads everything twice,
  and Qwen3.6-27B (52 GB) twice plus the ceselder NLA repo (69 GB) fills the 200 GB disk.
- RunPod containers see all the host's cores (96 on an H200 node) but get a ~10-core CPU quota.
  torch starts a thread per visible core, so CPU work (the O-lens whiteners' float64 `eigh`)
  thrashes: `pod/server.py` took 46+ min to load instead of 44 s. Start it with
  `OMP_NUM_THREADS=10`.
- vLLM's Qwen3.6 linear-attention kernels JIT-compile with `ninja`: put the venv's `bin/` on `PATH`,
  and expect ~10 min of compiling on the first run.
- If ssh to a pod drops or times out, check `runpodctl user` (balance) and `runpodctl pod list`
  before retrying: at a balance ≤ 0 RunPod deletes every pod on the account.
- For O-lens reads on a pod, follow `olens/README.md` (deps, `scripts/download.sh`,
  `PYTHONPATH=src`). Qwen3.6-27B in bf16 is ~54 GB, so it needs an 80 GB+ GPU.
