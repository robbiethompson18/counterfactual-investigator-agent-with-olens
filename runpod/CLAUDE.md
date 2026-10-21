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

## Shared Qwen3.6-27B server: use it, don't start your own

One vLLM server serves Qwen3.6-27B chat completions to every agent: pod `qwen-shared`, 2x H200
($10.58/hr), one vLLM replica per GPU behind one OpenAI-compatible endpoint. Before it existed,
three or four agents each ran their own mostly idle vLLM pod and each rediscovered the same startup
bugs.

- **The one client standard:** Python code samples Qwen via
  `counterfactual_investigator_agent_with_olens.qwen.client()`. It reads `$QWEN_VLLM_URL` and
  `$QWEN_VLLM_API_KEY` from `.envrc.local` and sets the TCP keepalives and long timeout every client
  needs. Non-Python clients (e.g. the envs gateway in a container) take those same two env vars. No
  hardcoded `localhost:18000`/`28000` tunnels, replica lists or other URL env vars: vLLM balances
  its replicas behind one URL.
- The model name is `qwen` (or `Qwen/Qwen3.6-27B`). The server has 262k context, parses reasoning
  (`reasoning_content`) and tool calls (`qwen3_coder`), and uses prefix caching. Sampling is
  per-experiment: pass it per request. The server default is Qwen's generation config.
- Check it's up with `runpod/qwen_server.sh status`. If it's down, bring it up with
  `runpod/qwen_server.sh up`, then write `runpod/qwen_server.sh url` into `QWEN_VLLM_URL` in
  `.envrc.local`. Every checkout symlinks to the main checkout's `.envrc.local`, so they all pick up
  the new URL. The IP and port change whenever the pod is recreated.
- It serves the base model only. O-lens reads (hooked forward passes) and LoRA model organisms still
  need their own pods.
- It uses a raw TCP port, not RunPod's `-8000.proxy.runpod.net` HTTP proxy, which cuts requests off
  at about 100 s.
- Long non-streaming requests go silent for minutes while vLLM generates, and something on the path
  drops idle connections. That's why the client sets keepalives. A raw client must set them too, or
  stream.

### Bringing up any pod: check progress every 3 minutes

Agents keep starting a pod install, watching only for "ready" or an error, and then sitting on a
hang for half an hour (a uv download once stalled for 26 min, silently). While anything is booting,
run a progress check every 3 minutes (a Monitor loop or cron), e.g. `runpod/qwen_server.sh status`.
If two checks in a row print the same line, it's stuck: look at it, kill it, and retry. Don't keep
waiting. Watch for READY, errors, and no change, not only READY.

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
- vLLM's Qwen3.6 linear-attention kernels JIT-compile with `ninja`: put the venv's `bin/` and
  `/usr/local/cuda/bin` on `PATH`, and expect ~10 min of compiling on the first run.
- `pkill -f PATTERN` over ssh kills your own remote shell if PATTERN appears anywhere in the ssh
  command (e.g. you also `cat > PATTERN.sh` in it). Use a `[x]yz` pattern, or a separate ssh call.
- `nohup bash -s ... &` over ssh reads an empty stdin, so the piped script never runs. Copy the
  script to the pod first, then `nohup bash /workspace/script.sh &`.
- If ssh to a pod drops or times out, check `runpodctl user` (balance) and `runpodctl pod list`
  before retrying: at a balance ≤ 0 RunPod deletes every pod on the account.
- For O-lens reads on a pod, follow `olens/README.md` (deps, `scripts/download.sh`,
  `PYTHONPATH=src`). Qwen3.6-27B in bf16 is ~54 GB, so it needs an 80 GB+ GPU.
