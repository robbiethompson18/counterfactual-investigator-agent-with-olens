# runpod/: Robbie's GPU pods

Rules for agents working for Robbie. The reference behind them (boot recipe, volumes, measurements,
every gotcha, S3 access) is [docs/runpod-qwen-server.md](../docs/runpod-qwen-server.md); read it
when something misbehaves, not before every command.

Robbie and Kylie keep their GPU compute separate:

- **Robbie** (and his agents) use plain RunPod pods on his account, via `runpodctl` and the scripts
  here. Never SkyPilot (`sky`, `transcript-viewer/sky/`), never the bramble GPU, and never anything
  that launches GPUs from bramble (e.g. the bramble:8801 lens explorer's Start GPU button or its
  local backend). Those are Kylie's accounts and hardware. Reading her viewers and caches is fine.
- **Kylie** (and her agents) use SkyPilot and bramble. If you're working for Kylie, this directory
  isn't for you.

## Shared Qwen3.6-27B server: use it, don't start your own

One vLLM server (pod `qwen-shared`, 2x H200, ~$10.58/hr, one replica per GPU behind one endpoint)
serves Qwen3.6-27B chat completions to every agent. Before it existed, three or four agents each ran
their own mostly idle pod and each rediscovered the same startup bugs.

- **One client.** Python samples Qwen via
  `counterfactual_investigator_agent_with_olens.qwen.client()`. It reads `$QWEN_VLLM_URL` and
  `$QWEN_VLLM_API_KEY` from `.envrc.local` and sets the TCP keepalives and long timeout every client
  needs. Non-Python clients take those same two env vars. No hardcoded tunnels, replica lists or
  other URL env vars.
- Model name `qwen` (or `Qwen/Qwen3.6-27B`); 262k context; reasoning parsed into
  `reasoning_content`; tool calls parsed (`qwen3_coder`); prefix caching on. Sampling is per
  request; the server default is Qwen's generation config.
- `runpod/qwen_server.sh status` says whether it's up. If not: `runpod/qwen_server.sh up`, then
  write `runpod/qwen_server.sh url` into `QWEN_VLLM_URL` in `.envrc.local` (every checkout symlinks
  to the main checkout's). The IP and port change whenever the pod is recreated.
- It serves the base model only. O-lens reads and LoRA organisms need their own pod (below).
- **Set `max_tokens` on thinking requests** (e.g. 65536). Uncapped generations that loop fill the KV
  cache and stall the whole server.
- **Keep concurrency where KV cache stays below ~85%.** At 100% every turn of a multi-turn agent
  reprocesses its whole history.
- **Killing a client doesn't free the server.** RunPod's proxy hides client disconnects from vLLM,
  so after a mass kill-and-relaunch, restart vLLM to flush the queue if its running + waiting is far
  above your open connections.
- `qwen_server.sh status` prints SM clock, temperature and throttle bits per GPU. Under load, 0x20
  or 0x40 means a thermally throttled host (decode ran 3x slow once); replace the pod.

## Any pod: watch it, then delete it

- **While anything boots, check progress every 3 minutes** (`qwen_server.sh status`, or the job's
  log), with a Monitor loop or cron. Two identical checks in a row means stuck: look, kill, retry.
  Agents have sat on a silent hang for half an hour waiting for "ready".
- An H200 is about $5/hr. Delete pods the moment a job finishes (`runpod/pod.sh down NAME`).
- At a balance ≤ 0 RunPod deletes every pod on the account; if ssh drops, check `runpodctl user`.

## Scripts

Each script's header has its usage.

- `qwen_server.sh up|url|status|restart`: the shared server, booted in ~3 min from the vLLM image
  plus a network volume that caches weights and compiles. `NAME=x ... up 1` boots a one-GPU pod on
  the same recipe for other jobs.
- `olens_reads.sh setup|run|status|fetch|stop NAME ...`: batch O-lens reads (agu18dec space lens +
  AR, `scripts/olens_read_requests.py`) of whole transcripts on a pod from
  `NAME=x qwen_server.sh up 1`. Write outputs as a `scripts/runs.py` run under
  `data/runs/olens_reads/`.
- `pod.sh up|ssh|sync|down|list NAME`: bare `runpod/pytorch` pods for anything else. Needs
  `RUNPOD_API_KEY` (`.envrc.local`) and a logged-in `hf` CLI.
- Logs, reads and checkpoints on the two volumes can be read with no pod running, over RunPod's S3
  API (`aws s3 ls s3://<volume id>/ --profile runpod-ca2|runpod-nc1`; setup in the doc).
