# WeirdChat × oracle lens viewer

Browse the Qwen3.6-27B patterns of
[Transluce/WeirdChat](https://huggingface.co/datasets/Transluce/WeirdChat) (225 patterns, 25,792
transcripts) token by token, and read any token with the agu18dec oracle lens (`../olens`). Reads
run on cloud GPUs and are cached locally, so everything read once stays browsable with no GPU up.

## What the page does

- Transcripts rendered through Qwen3.6-27B's chat template (`enable_thinking=False`); judge
  citations highlighted; end-of-turn zones (`<|im_end|>` through the next message's content) marked,
  since the lens reads best there.
- Click a token → readouts at the chosen layers (20..60 step 4): bullets sorted by their own FVE,
  the joint FVE (NNLS refit of all bullets' AR directions in whitened space), NNLS weights.
- FVE heatmap over the transcript (joint or top-bullet, any layer, per-transcript 5–95th percentile
  colour scale).
- A shared read queue: "Queue all relevant tokens" puts every end-of-turn and punctuation position
  of a transcript at the chosen layers at the front; the header pill opens the queue panel.

## Model organisms

`web/organisms.py` appends one pattern per LoRA model organism after WeirdChat's 225 (behavior
"model organism"), from `data/runs/organisms/` at the repo root (HF dataset `runs/organisms/`; setup
and findings in `docs/model-organisms.md`). Each rollout is a transcript. The ones read with the
lens (sample 1 of each trigger and control prompt) have reads on the organism's activations and on
base Qwen's over the same tokens: the transcript header switches between them, and the other one
shows under each readout. Base Qwen's rollouts on the same prompt are at the bottom. The viewer
can't make new reads of these (the pod doesn't load LoRAs; `scripts/organism_olens.py` does), and
the current reads have no FVE scores.

## Run

```bash
uv venv .venv && VIRTUAL_ENV=.venv uv pip install transformers==5.16.1 tokenizers==0.23.1 pyarrow==25.0.1 jinja2
hf download Transluce/WeirdChat --repo-type dataset --local-dir data/hf \
  --include 'data/patterns.parquet' 'data/rubrics.parquet' 'data/prompts/qwen__qwen3.6-27b/*' 'data/transcripts/qwen__qwen3.6-27b/*'
WCO_BIND="127.0.0.1" HF_TOKEN=... web/run.sh     # http://127.0.0.1:8803/
```

- `WCO_BIND`: space-separated addresses to listen on (e.g. a tailnet v4 and v6).
- `HF_TOKEN` (or `HF_TOKEN_FILE`): needed by GPU nodes for the gated whiteners.
- Reads are cached in `data/reads.jsonl`, the queue in `data/queue.json` (both gitignored).

## Sharing: the HF dataset and the public snapshot

[weirdchat-olens.pages.dev](https://weirdchat-olens.pages.dev) is this page in read-only mode (no GPU or queue
controls), with links in the same `#p=…&t=…` form. Its data lives in the project's HF dataset under `weirdchat/`, where
each machine owns its own files so publishes never clash:

- `weirdchat/reads/<machine>.jsonl`, `rollouts/<machine>.jsonl`, `stance/<machine>.jsonl`: copies of this machine's
  `data/reads.jsonl`, `data/claude.jsonl` and `data/stance.jsonl`. The machine name is `$WCO_MACHINE`, else the short
  hostname.
- `weirdchat/site/`: one gzipped JSON per transcript (the server's `/api/transcript`), per pattern, and the pattern
  list, built from everyone's files. The page fetches these straight from HF (CORS is open).

```bash
.venv/bin/python scripts/publish.py          # push this machine's files, pull everyone's, rebuild + upload the site
.venv/bin/python scripts/publish.py --pull   # only pull other machines' files, to browse them in the local viewer
scripts/deploy_site.sh                       # (from the repo root) only when web/index.html changes
```

The local server reads the pulled files (`data/shared/weirdchat/`) alongside its own. A full publish takes a few
minutes (it renders all 25,792 transcripts) and uploads only the site files that changed. Writing to the dataset
needs an HF token with write access to it.

## GPUs

- **Start GPU** in the header launches `sky/pod.yaml` (`web/pod.py`) with SkyPilot; the pod's server
  exits after 45 idle minutes (hard cap 3.75 h) and SkyPilot downs it 5 minutes later.
- **Bulk precompute**: `scripts/fleet.py` brings up several nodes (`sky/fleet*.yaml`; multi-GPU
  nodes run one server per GPU) that pull chunks from the viewer's queue over HTTP
  (`WCO_VIEWER_URL`, default `http://127.0.0.1:8803`), write to `data/reads.jsonl`, and tear
  themselves down after 5 idle minutes. Example:
  `scripts/fleet.py --ids 1,2 --multi-ids 2 --yaml-for 2:sky/fleet_h200x2.yaml --layers 24,32,40,48,56`.
- `--local` + `web/run_lens.sh`: read on this machine's GPU instead (~57 GB).

## Gotchas

- 80–85 GB cards run out of memory on long transcripts at big batches: they get batch 24 and only
  transcripts ≤ 1536 tokens from the queue; ≥ 90 GB cards take everything at batch 192.
- Vast "H100" offers often come up as other cards (32 GB RTX 5090s included): the fleet checks GPU
  memory and replaces nodes under 80 GB. RunPod is tried first.
- `sky exec` without `--gpus` hides the GPU from the job (`CUDA_VISIBLE_DEVICES` empty);
  `pod/run.sh` and `pod/fleet_job.sh` unset it.
- The SkyPilot API server runs launches one at a time; restarting a driver mid-launch leaves the old
  request queued. `fleet.py` adopts existing clusters, and a node's launch job counts as its server
  (no duplicate `sky exec`).
- AR quality grows with depth (held-out FVE 0.12 at L20 → 0.34 at L60), so FVE isn't comparable
  across layers.
