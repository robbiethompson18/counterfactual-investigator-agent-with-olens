# runpod/: Robbie's GPU pods

Rules for agents working for Robbie. The reference behind them (boot recipe, volumes, measurements,
every gotcha, S3 access) is [docs/runpod-qwen-server.md](../docs/runpod-qwen-server.md); read it
when something misbehaves, not before every command.

Robbie and Kylie keep their GPU compute separate:

- **Robbie** (and his agents) use plain RunPod pods on his account, via `runpodctl` and the scripts
  here. Never SkyPilot (`sky`), never the bramble GPU, and never anything that launches GPUs from
  bramble (e.g. the bramble:8801 lens explorer's Start GPU button or its local backend). Those are
  Kylie's accounts and hardware. Reading her viewers and caches is fine.
- **Kylie** (and her agents) use SkyPilot and bramble. If you're working for Kylie, this directory
  isn't for you.

## Cost guards (2026-10-25: eight idle H200s ran the balance to −$0.57, ~$220 wasted)

Eight read pods were created at 23:43 PDT on 2026-10-24 and the session went quiet before `setup` or
`run` ever reached them; they idled 7.7 h until RunPod deleted them at zero balance. Rules:

- **No pod without a live watcher.** Before `up`, start (and verify with `pgrep`) a detached process
  that deletes the pod when its job is done, and deletes everything if `clientBalance` falls under a
  floor (GraphQL `{ myself { clientBalance } }`). A Claude session is not a watcher: it dies with
  its process tree. The 10-minute wakeup loop in Robbie's `CLAUDE.local.md` is the second layer.
- **Never `--watch` a reader unless that watcher is running.** `--watch` keeps the GPU busy-idle
  forever.
- **Fetch as you go.** Reads that exist only on a pod or a network volume don't exist: pull them
  while the pod is up, not at the end.
- **Poll the REST API, not runpodctl, for readiness:** `GET /v1/pods/<id>` → `publicIp` and
  `portMappings["22"]`; then `ssh -o ConnectTimeout=8` wrapped in a hard timeout
  (`perl -e 'alarm 20; exec @ARGV' ssh ...`; macOS has no `timeout`). `runpodctl pod get` reports
  `runtime: null` for running pods, and with an empty id it blocks on stdin.
- **A pod can be RUNNING and billed with no machine behind it** (no `publicIp` for 9 min on
  US-CA-2). Give a host 5 min to show ports and 15 min to answer ssh (cold hosts pull the 15 GB vLLM
  image), then delete it and try the other volume's datacenter.
- **Small container disks kill pods silently.** `runpod/pytorch` needs more than 20 GB (`pod.sh`
  uses 200); pods created with 20 GB vanished within 2 min and left a 404 on delete.
- Keep the balance above ~1 h of the fleet's burn; at zero RunPod deletes the pods (audit log actor
  `runpod`). The volumes survive, and need no GPU to read (S3 API or a $0.06/h CPU pod; the doc has
  both recipes).

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
- **Pods on one network volume share `/workspace`, so every reader on it needs its own TAG**
  (`olens_reads.sh run NAME REQ TAG`): two pods given the same tag append to one
  `/workspace/reads/TAG.reads.jsonl` and overwrite each other's log (2026-10-29, the supergpqa-3k
  O-lens shards, caught when both fetches came back byte-identical). Tag per pod, e.g. `sg3k_ol-1`.
- An H200 is about $5/hr. Delete pods the moment a job finishes (`runpod/pod.sh down NAME`).
- At a balance ≤ 0 RunPod deletes every pod on the account; if ssh drops, check `runpodctl user`.

## Scripts

Each script's header has its usage.

- `qwen_server.sh up|url|status|restart`: the shared server, booted in ~3 min from the vLLM image
  plus a network volume that caches weights and compiles. `NAME=x ... up 1` boots a one-GPU pod on
  the same recipe for other jobs.
- `olens_reads.sh setup|run|status|fetch|stop NAME ...`: batch O-lens reads (agu18dec space lens +
  AR, `scripts/olens_read_requests.py`; capture, lens and AR on one vLLM,
  [docs/olens-one-engine-reader.md](../docs/olens-one-engine-reader.md)) of whole transcripts on a
  pod from `IMAGE=vllm/vllm-openai:v0.27.1 NAME=x qwen_server.sh up 1`. Write outputs as a
  `scripts/runs.py` run under `data/runs/olens_reads/`. `fetch` also copies `reads.provenance.json`
  (which lens, bullet prefix, AR, whitener, checkpoint revisions; `olens/src/olens/provenance.py`)
  next to the reads: keep it with them, it is the only machine-written record of which lens a batch
  used.
- `nla_reads.sh setup|run|status|fetch|stop|score NAME ...`: the same reader with `--readout nla`
  (the all-layer NLA's AV on the one engine, `olens/src/olens/nla.py`), writing `TAG.reads.jsonl`
  plus `TAG.reads.acts.bf16` (each decoded read's activation); `score` then runs the NLA critic
  (`scripts/nla_score.py`, a separate 51 GB model, transformers) on the pod to add `fve` and `cos`
  to the rows. Runs go under `data/runs/nla_reads/` ([docs/nla-reader.md](../docs/nla-reader.md)).
- `pod.sh up|ssh|sync|down|list NAME`: bare `runpod/pytorch` pods for anything else. Needs
  `RUNPOD_API_KEY` (`.envrc.local`) and a logged-in `hf` CLI.
- Logs, reads and checkpoints on the two volumes can be read with no pod running, over RunPod's S3
  API (`aws s3 ls s3://<volume id>/ --profile runpod-ca2|runpod-nc1`; setup in the doc).
