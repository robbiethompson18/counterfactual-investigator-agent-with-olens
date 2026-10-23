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
  need their own pods: see "O-lens reads" below. vLLM's LoRA runtime does work on this model once
  the adapter keys are renamed ("vLLM LoRA" below), so the shared server could serve them too.
- Set `max_tokens` on thinking-mode requests (e.g. 65536). On 2026-10-21 a handful of uncapped Qwen
  generations looped in their thinking, filled one engine's KV cache, and the server then finished
  ~0 requests/min while preempting (6 restarts that night). No normal finished generation went past
  50k tokens, so a 64k cap only cuts loops.
- For multi-turn agents, keep concurrency where KV cache stays below ~85%. At 100%, each episode's
  prefix is evicted between turns and every turn reprocesses the whole history (6% prefix-cache hit
  vs 56% on a server with headroom).
- Clients need TCP keepalives (`qwen.client()` sets them). Without them a long silent generation's
  connection is dropped on the way to RunPod, vLLM's response goes nowhere, and the client hangs
  forever.
- Killing a client doesn't free the server. RunPod's TCP proxy keeps the server side of a connection
  open after the client dies, so vLLM never sees the disconnect and keeps generating for queued
  requests from killed runs (7.3k of them on 2026-10-21). After a mass kill-and-relaunch, count
  client connections (`lsof -nP -iTCP:PORT -sTCP:ESTABLISHED`); if they're far below the server's
  running + waiting, restart vLLM to flush the queue.
- It uses a raw TCP port, not RunPod's `-8000.proxy.runpod.net` HTTP proxy, which cuts requests off
  at about 100 s.
- Long non-streaming requests go silent for minutes while vLLM generates, and something on the path
  drops idle connections. That's why the client sets keepalives. A raw client must set them too, or
  stream.

### vLLM version: 0.32 matches 0.25.1; the "3x slower" was most likely a throttled GPU

The live `qwen-shared` pod is the older pip-installed vLLM 0.25.1 recipe (runpod/pytorch image, venv
in `/workspace/vllm-env`), not the image below. A first 0.32 test on 2026-10-21 decoded about 3x
slower per GPU than 0.25.1. The A/B below says that came from the host, not the version: on healthy
H200s, `vllm/vllm-openai:v0.32.0` matches 0.25.1 within 3%, including on the US-CA-2 volume with
`HOME` on NFS.

Output tok/s per GPU (1x H200 unless noted), `vllm bench serve` run on the pod (random 256 in / 1500
out, `--ignore-eos`, 2×concurrency prompts), the shared server's flags:

| Setup                                         | 32/GPU | 128/GPU | 256/GPU |
| --------------------------------------------- | ------ | ------- | ------- |
| v0.25.1 image, local disk                     | 1,621  | 3,940   | 4,692   |
| v0.32.0 image, local disk                     | 1,677  | 4,022   | 4,596   |
| v0.32.0 image, US-CA-2 volume (`up` recipe)   | 1,664  | 3,972   | 4,565   |
| v0.32.0 image, 2x H200 data parallel, per GPU | 1,672  | 3,850   | 4,180   |
| v0.25.1 image, thermally throttled GPU        | 654    |         |         |

The repo's chat load test (non-thinking essays, max_tokens 1500) agrees: 0.25.1 gives 1,664 / 4,173
/ 5,028 tok/s on the pod, and 0.32 on the volume gives 1,690 / 4,103 / 4,876 on the pod and 1,686 /
4,152 / 4,896 from a Mac over the internet. So neither the NFS `HOME` nor the client path costs
anything.

The slow pod in this run (EUR-IS-4) had its SM clock pinned at 345 MHz under load, at 92-94 C, with
throttle reason 0x20 (SW thermal slowdown) and 6 h of throttling on its driver counters. That's 21
tok/s per stream at 32 concurrent, the same as the original "slow 0.32" (25). The original pod's
logs were overwritten, so its host can't be checked now. A healthy H200 under load sits at
1,500-1,980 MHz, under 80 C, with reasons 0x4 (power cap) or 0. `nvidia-smi`'s max clock reads 1980
MHz on a throttled GPU too, and an idle GPU reads 345 MHz with 0x1, so only clocks under load tell.
`qwen_server.sh status` now prints clocks, temperature and throttle bits per GPU; check them while
it has traffic, and replace the pod if it shows 0x20/0x40.

Before calling a version or flag slow, compare it on the same host, or log
`nvidia-smi --query-gpu=clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv -l 5`
during each run.

Don't run `runpod/qwen_server.sh restart` on the live `qwen-shared` pod: the boot script now expects
the vllm image. To restart vLLM there, ssh in and run its own
`API_KEY=... nohup bash /workspace/qwen_server_boot.sh > /workspace/boot.log 2>&1 < /dev/null &`.

### How it boots in ~3 minutes

`up` creates the pod through RunPod's REST API (runpodctl can't set a start command) from the
official `vllm/vllm-openai:v0.32.0` image. The pod mounts a network volume at `/workspace`. There
are two, each 150 GB (about $10.50/month), and `up` tries them in order until one's datacenter has
the GPUs free: `qwen-fastboot` (id `wwwce2mejb`, **US-CA-2**), then `qwen-fastboot-nc1` (id
`cdde9ia5iu`, **US-NC-1**, added 2026-10-22 when US-CA-2 ran out of H200s). `VOLUMES=<id>` picks
one. The image's start command is `runpod/qwen_server_boot.sh`, which installs sshd and starts vLLM.
The volume holds the weights (`HF_HOME`) and every compile cache: the boot script sets
`HOME=/workspace/home`, and vLLM's torch.compile/inductor cache, FlashInfer, Triton etc. all write
under `~`. So only the first boot on a fresh volume downloads and compiles.

Measured 2026-10-21 on 1x H200 (2x H200 had no stock in any datacenter with network volumes):

| Step                                                | Cached boot | First boot, fresh volume |
| --------------------------------------------------- | ----------- | ------------------------ |
| `up` to container start (image already on the host) | 8 s         | 8 s                      |
| sshd (from `.deb`s cached on the volume)            | 4 s         | 25 s (`apt-get`)         |
| Weights                                             | 1 s (check) | 1 min 50 s (52 GB)       |
| vLLM imports, engine start, weight load (20 s)      | 75 s        | 100 s                    |
| torch.compile (incl. Dynamo tracing)                | 18 s        | 97 s                     |
| Warmup + CUDA graph capture                         | 63 s        | 95 s                     |
| **`up` to `/v1/models` answering**                  | **3 min**   | about 7.5 min            |

The old recipe (pip-install vLLM, then JIT-compile kernels) took 30 to 45 min.

Gotchas:

- **Image pull is the one uncached step.** A host that doesn't already have the image pulls it in 9
  to 10 min, so `up` then takes about 13 min. A volume pins the pod to one datacenter, which has
  only a few H200 hosts, so after the first pull on a host later boots there are fast.
- **A volume pins the pod to its datacenter.** If no volume's datacenter has free H200s, `up` fails
  with a REST error. `up 1` asks for one GPU, which is likelier to be free. Another datacenter means
  a new volume (REST `POST /v1/networkvolumes`, then add its id to `VOLUMES`) and one first boot to
  fill it (we know of no way to copy a volume across datacenters). Check H200 stock per datacenter,
  and which take volumes, with the GraphQL
  `dataCenters { id storageSupport gpuAvailability { gpuTypeId stockStatus } }` query. On 2026-10-22
  H200s were in CA-MTL-3, EUR-IS-4, US-CO-1 and US-NC-1 (with volumes) and EUR-IS-5, US-GA-2
  (without); skip EUR-IS-4, home of the throttled host.
- The first boot on `qwen-fastboot-nc1` (2026-10-22) took 3.5 min to READY, weights included (52 GB
  in 25 s), on a host that already had the image.
- **2 GPUs on the volume: not tested,** because US-CA-2 had no 2x H200 stock (again on 2026-10-21).
  The 0.32 image with 2x data parallel works off the volume (table above). The compile cache lives
  in per-rank directories (`rank_0_0`, `rank_0_1`), so the first 2-GPU boot on the volume will
  compile once more (about 80 s).
- **Use the default `v0.32.0` tag (CUDA 13, needs host driver 580+), not `v0.32.0-cu129`.** The
  cu129 image ships a CUDA 13 build of torchcodec, and vLLM dies at import with
  `libnvrtc.so.13: cannot open shared object file`.
- **The vllm image has no sshd.** The boot script installs `openssh-server` from `.deb`s cached on
  the volume, one directory per Ubuntu release (`/workspace/debs-24.04`). Debs from a 22.04 image
  broke `dpkg` on the 24.04 one, which left the container in a restart loop with no way in.
- **The image ships prebuilt FlashInfer kernels** (`flashinfer-jit-cache`), so the 9 min FlashInfer
  and linear-attention JIT compile from the pip recipe doesn't happen. Nothing needs `ninja`.
- **Don't put Python packages or `__pycache__` on the volume.** It's NFS: reads are fast in bulk (52
  GB of weights load in 20 s) but slow per small file. Importing vLLM took 45 s with bytecode on the
  volume vs 11 s on container disk.
- `restart` runs the boot script over ssh, whose env lacks the image's `ENV` (`CUDA_HOME`,
  `TORCH_CUDA_ARCH_LIST`, ...) and the pod's (`API_KEY`). The script imports PID 1's env first.
  Without that, a restart missed vLLM's compile cache and recompiled (76 s).
- The start command is `bash -c <boot script>`, so PID 1's command line contains the whole script.
  `pgrep -f` patterns in it need the `[b]racket` trick or they match PID 1.
- `/workspace/boot.log` and `/workspace/vllm.log` are on the volume, so they survive the pod.
  `vllm.log` prints the API key in its arguments line; don't paste it anywhere.

### Bringing up any pod: check progress every 3 minutes

Agents keep starting a pod install, watching only for "ready" or an error, and then sitting on a
hang for half an hour (a uv download once stalled for 26 min, silently). While anything is booting,
run a progress check every 3 minutes (a Monitor loop or cron), e.g. `runpod/qwen_server.sh status`.
If two checks in a row print the same line, it's stuck: look at it, kill it, and retry. Don't keep
waiting. Watch for READY, errors, and no change, not only READY.

## O-lens reads: `runpod/olens_reads.sh`

Batch O-lens reads of whole transcripts, for any agent that wants them: the agu18dec lens + AR
(`olens/`) at every stop token (the viewer's PUNCT regex plus every `<|im_end|>`, prompt included),
greedy, 4 bullets, per-bullet and joint whitened FVE. `scripts/olens_read_requests.py` does the
work; its docstring has the request and output formats. The lens is decoded by vLLM from merged
weights (`olens.vllm_lens`); the HF base on the same GPU captures activations and runs the AR.

```
NAME=lens-reads runpod/qwen_server.sh up 1            # vLLM image + cache volume, ~3.5 min
runpod/olens_reads.sh setup lens-reads                # stops its vLLM server; deps, checkpoints, merged lens
runpod/olens_reads.sh run lens-reads REQ.jsonl TAG --layers 24,32,40,44,48,56 [--shard 0/2]
runpod/olens_reads.sh status lens-reads TAG
runpod/olens_reads.sh fetch lens-reads TAG data/runs/olens_reads/<run>/reads.jsonl
runpod/olens_reads.sh stop lens-reads                 # rerunning `run` with the same TAG resumes
```

Write the outputs as a `scripts/runs.py` run under `data/runs/olens_reads/`. First used for the
cfbench spike set (2026-10-22, 392 env transcripts, 1.16M tokens, 84k stop tokens).

Gotchas:

- **One GPU holds two copies of Qwen.** vLLM gets `--vllm-mem 0.55` (default) and HF the rest (~63
  GB; it peaks at ~52 GB). Each concurrent decode needs one Mamba (linear-attention state) block:
  0.45 of an H200 fits only 139, so vLLM refuses `max_num_seqs` 512; 0.55 fits 256.
- **Capture is chunked** (4096 tokens through the cache): one forward over a 15k-token transcript
  ran out of memory next to vLLM. Chunking moves the residual by bf16 noise (256-token chunks vs one
  forward: mean cos 0.997-0.99998, min 0.958 at L56); vLLM prefills in chunks too.
- **`olens.ar.AR` must run with `use_cache=False`** (fixed 2026-10-22): with the default cache a
  256-span AR batch also built every layer's linear-attention state and ran out of memory.
- **Killing the reader orphans `VLLM::EngineCore`,** which keeps ~75 GB of GPU memory, and the next
  run fails at engine start. `olens_reads.sh stop` kills both.
- **A vLLM still loading weights can ignore SIGTERM;** `setup` waits and sends SIGKILL.
- The merged lens (52 GB) lives on the cache volume (`/workspace/merged/agu_rl600`); `setup` builds
  it once per volume (~10 min). A volume needs ~110 GB free for weights + merged lens.
- The vllm image lacks scipy, pyyaml and flash-linear-attention; `setup` pip-installs them into the
  image's Python (not the volume: NFS makes imports slow). HF also warns that `causal_conv1d` is
  missing and falls back to PyTorch; capture is still ~1 s per 1k tokens.

## vLLM LoRA on Qwen3.6-27B: works once the adapter keys are renamed

Agam found vLLM's LoRA runtime "inert" on this model, which is why lenses and organisms get merged
into full 52 GB copies. The cause is key names. The adapters were trained on HF's text-only model
(`base_model.model.model.layers.N...`); vLLM loads the checkpoint as
`Qwen3_5ForConditionalGeneration`, whose mapper only rewrites `model.language_model.` prefixes, so
no module gets its weights. vLLM 0.32 raises nothing, and its DEBUG log says "Successfully loaded
LoRA weights" for every module anyway. `olens/scripts/lora_for_vllm.py` renames them to
`model.language_model.layers.N...`.

Measured 2026-10-22 on 1x H200, vLLM 0.32, 792 greedy lens reads (6 chat prompts x 12 end-of-turn
positions x 11 layers, `prompt_embeds`), compared with the merged-weights lens:

| Arm                        | Exact text | First line | First token | reads/s |
| -------------------------- | ---------- | ---------- | ----------- | ------- |
| base Qwen (no LoRA)        | 0%         | 0%         | 1%          | 30.5    |
| LoRA, raw keys             | 0%         | 0%         | 1%          | 25.0    |
| LoRA, renamed keys         | 40%        | 87%        | 100%        | 33.5    |
| merged weights (reference) | 100%       | 100%       | 100%        | 41.2    |

Raw keys also match base Qwen's text exactly on 430/792: the adapter does nothing. The renamed
LoRA's misses are bf16 noise (HF vs vLLM-merged agree on ~80% of first bullets,
`olens/vllm_lens.py`). LoRA decodes ~19% slower than merged weights here.

So the shared server could serve the lens (and organisms) as adapters:
`--enable-lora --max-lora-rank 16 --lora-modules olens=<renamed dir> --enable-prompt-embeds`. vLLM
0.32's prefix cache hashes prompt embeds per block, so that should keep the agents' prefix-cache
hits. Not done yet: we haven't measured what `--enable-lora` costs base-model chat throughput, and
the readout still needs an HF process for capture and the AR.

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
