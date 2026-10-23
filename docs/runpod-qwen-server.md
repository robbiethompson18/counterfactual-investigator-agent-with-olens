# The shared Qwen3.6-27B server and RunPod pods: how it works

**Status (2026-10-26):** ACTIVE. The recipe `runpod/qwen_server.sh up` uses, its measurements and
gotchas, the volumes' S3 access, and the generic `pod.sh` gotchas. The rules an agent must follow
are in [runpod/CLAUDE.md](../runpod/CLAUDE.md); this is the reference behind them.

## How `qwen_server.sh up` boots in ~3 minutes

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

The old recipe (pip-install vLLM into a `runpod/pytorch` image, then JIT-compile kernels) took 30 to
45 min. It was retired on 2026-10-22; every pod since uses the image recipe.

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
  (without); skip EUR-IS-4, home of the throttled host (below).
- The first boot on `qwen-fastboot-nc1` (2026-10-22) took 3.5 min to READY, weights included (52 GB
  in 25 s), on a host that already had the image.
- **2 GPUs on the volume: not tested,** because US-CA-2 had no 2x H200 stock (again on 2026-10-21).
  The 0.32 image with 2x data parallel works off the volume (table below). The compile cache lives
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

## Why the server is configured the way it is (lessons from the 2026-10-21 overnight sweep)

- **`max_tokens` on thinking requests.** A handful of uncapped Qwen generations looped in their
  thinking, filled one engine's KV cache, and the server then finished ~0 requests/min while
  preempting (6 restarts that night). No normal finished generation went past 50k tokens, so a 64k
  cap only cuts loops.
- **KV headroom.** At 100% KV cache each episode's prefix is evicted between turns and every turn
  reprocesses the whole history: 6% prefix-cache hit vs 56% on a server with headroom. Keep
  concurrency where the cache stays below ~85%.
- **Keepalives.** Long non-streaming requests go silent for minutes while vLLM generates, and
  something on the path to RunPod drops idle connections. Without TCP keepalives vLLM's response
  goes nowhere and the client hangs forever. `qwen.client()` sets them; a raw client must too, or
  stream.
- **Killing a client doesn't free the server.** RunPod's TCP proxy keeps the server side of a
  connection open after the client dies, so vLLM never sees the disconnect and keeps generating for
  queued requests from killed runs (7.3k of them on 2026-10-21). After a mass kill-and-relaunch,
  count client connections (`lsof -nP -iTCP:PORT -sTCP:ESTABLISHED`); if they're far below the
  server's running + waiting, restart vLLM to flush the queue.
- **Raw TCP port, not RunPod's `-8000.proxy.runpod.net` HTTP proxy,** which cuts requests off at
  about 100 s.

## vLLM 0.32 matches 0.25.1; the "3x slower" was a thermally throttled GPU

A first 0.32 test on 2026-10-21 decoded about 3x slower per GPU than the pip-installed 0.25.1. The
A/B below says that came from the host, not the version: on healthy H200s,
`vllm/vllm-openai:v0.32.0` matches 0.25.1 within 3%, including on the US-CA-2 volume with `HOME` on
NFS.

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
`qwen_server.sh status` prints clocks, temperature and throttle bits per GPU; check them while it
has traffic, and replace the pod if it shows 0x20/0x40.

Before calling a version or flag slow, compare it on the same host, or log
`nvidia-smi --query-gpu=clocks.sm,temperature.gpu,clocks_throttle_reasons.active --format=csv -l 5`
during each run.

## O-lens reads on a pod (`runpod/olens_reads.sh`): gotchas

The usage is in the script's header and in `runpod/CLAUDE.md`. `scripts/olens_read_requests.py` does
the work; its docstring has the request and output formats. First used for the cfbench spike set
(2026-10-22, 392 env transcripts, 1.16M tokens, 84k stop tokens).

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
- **Don't run `setup` the moment ssh answers.** The boot script starts vLLM a few seconds after
  sshd. On 2026-10-23 `setup` ran first, found no vLLM to kill, and the boot script then started one
  that held 127 GB, so the next job hit CUDA OOM. Wait for `vllm starting` in `/workspace/boot.log`
  before `setup`, or check `nvidia-smi` after it.
- The merged lens (52 GB) lives on the cache volume (`/workspace/merged/agu_space_rl600`); `setup`
  builds it once per volume (~10 min). A volume needs ~110 GB free for weights + merged lens.
- The vllm image lacks scipy, pyyaml and flash-linear-attention; `setup` pip-installs them into the
  image's Python (not the volume: NFS makes imports slow). HF also warns that `causal_conv1d` is
  missing and falls back to PyTorch; capture is still ~1 s per 1k tokens.

## Reading the volumes without a pod: RunPod's S3 API

Both network volumes are reachable over RunPod's S3-compatible API
([docs](https://docs.runpod.io/storage/s3-api)), so logs, reads and checkpoints can be listed or
copied with no pod running. Bucket = volume id; the endpoint and region are per datacenter, so
`~/.aws/config` has one profile per volume (set up 2026-10-26; the key lives in `~/.aws/credentials`
and, as `RUNPOD_S3_ACCESS_KEY` / `RUNPOD_S3_SECRET_ACCESS_KEY`, in `.envrc.local`):

| Profile      | Volume                           | Endpoint                           |
| ------------ | -------------------------------- | ---------------------------------- |
| `runpod-ca2` | `wwwce2mejb` (qwen-fastboot)     | `https://s3api-us-ca-2.runpod.io/` |
| `runpod-nc1` | `cdde9ia5iu` (qwen-fastboot-nc1) | `https://s3api-us-nc-1.runpod.io/` |

```
aws s3 ls s3://wwwce2mejb/reads/ --profile runpod-ca2
aws s3 cp s3://wwwce2mejb/vllm.log - --profile runpod-ca2 | tail
aws s3 cp local.jsonl s3://cdde9ia5iu/novelty_in/ --profile runpod-nc1
```

Keys are console-only (Credentials → S3 API Keys); neither the REST API nor `runpodctl` creates
them. On a new machine, re-create the profiles with the stanzas in Robbie's `~/.aws/config` (region
= datacenter id, `endpoint_url` as above, `multipart_chunksize = 256MB`) and
`aws configure set aws_{access_key_id,secret_access_key} ... --profile NAME`.

Gotchas:

- Single `PutObject` is capped under 500 MB; the CLI multiparts above `multipart_chunksize`, so big
  uploads work, but keep the chunk size under 500 MB.
- `ListObjects` is slow or fails ("same next token") the first time on a prefix with more than ~10k
  files or 10 GB, e.g. `.cache/huggingface/hub/`. Don't `aws s3 sync` the whole volume; list and
  copy specific prefixes.
- No batch `DeleteObjects`, presigned URLs, bucket create/delete or ACLs. Requests with the clock
  more than an hour off are rejected.

## Plain pods (`runpod/pod.sh`): gotchas

`pod.sh up` creates a bare `runpod/pytorch` pod with a 200 GB container disk and no volume; use it
for anything that isn't the Qwen server or O-lens reads. `PORTS=22/tcp,8000/tcp` exposes extra
ports.

- `runpodctl pod create --env` only reaches PID 1, not ssh sessions. `up` copies the `HF_*` vars
  into `/etc/environment` so `ssh` commands see `HF_TOKEN`.
- `sync` skips `.envrc.local`, `.venv`, `data/` and `checkpoints/`, so secrets and big files stay
  local. Don't copy `.envrc.local` to a pod.
- The image sets `HF_HOME=/workspace/.cache/huggingface` (and `up` copies it into
  `/etc/environment`). Don't set a different `HF_HOME` on the pod: it downloads everything twice,
  and Qwen3.6-27B (52 GB) twice plus the ceselder NLA repo (69 GB) fills the 200 GB disk.
- RunPod containers see all the host's cores (96 on an H200 node) but get a ~10-core CPU quota.
  torch starts a thread per visible core, so CPU work (the O-lens whiteners' float64 `eigh`)
  thrashes: `transcript-viewer/pod/server.py` took 46+ min to load instead of 44 s. Start CPU-heavy
  Python with `OMP_NUM_THREADS=10`.
- `pkill -f PATTERN` over ssh kills your own remote shell if PATTERN appears anywhere in the ssh
  command (e.g. you also `cat > PATTERN.sh` in it). Use a `[x]yz` pattern, or a separate ssh call.
- `nohup bash -s ... &` over ssh reads an empty stdin, so the piped script never runs. Copy the
  script to the pod first, then `nohup bash /workspace/script.sh &`.
- If ssh to a pod drops or times out, check `runpodctl user` (balance) and `runpodctl pod list`
  before retrying: at a balance ≤ 0 RunPod deletes every pod on the account.
- For O-lens reads on a plain pod, follow `olens/README.md` (deps, `olens/scripts/download.sh`,
  `PYTHONPATH=olens/src`). Qwen3.6-27B in bf16 is ~54 GB, so it needs an 80 GB+ GPU.
