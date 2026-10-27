# Transcript viewer

The read caches (`data/reads.jsonl`, `data/weirdchat/reads/*`) hold only old-lens
(`olens_s3d_rl600`) reads, tagged as such since 2026-10-23; the page shows them marked "old lens"
until the space lens re-reads a token, which was never scheduled
([docs/olens-space-migration.md](../docs/olens-space-migration.md)). The SkyPilot sections are
Kylie's setup; Robbie's agents use RunPod (`runpod/CLAUDE.md`).

Browse the Qwen3.6-27B patterns of
[Transluce/WeirdChat](https://huggingface.co/datasets/Transluce/WeirdChat) (225 patterns, 25,792
transcripts) and the 300 [CHIVE](https://huggingface.co/datasets/agu18dec/chive-forensics) incidents
token by token, and read any token with the agu18dec oracle lens (`../olens`). Reads run on cloud
GPUs and are cached locally, so everything read once stays browsable with no GPU up.

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

## CHIVE incidents

The CHIVE tab at the top of the list switches to CHIVE (`web/chive.py`). Each incident is one
pattern with one transcript, numbered after WeirdChat's 225 so old `#p=` links keep working.

- **Rendering:** from CHIVE's own `token_ids` and per-token turn labels, not our chat template,
  because Agam's banks index into those ids.
- **Not Qwen:** only the final reply is Qwen's. Earlier assistant turns come from the source dataset
  (mostly WildChat, written by ChatGPT), and the page labels them.
- **✂ cut off:** CHIVE sampled the final reply with a ~500-token cap. 176 of 300 hit it and stop
  mid-sentence, followed by an appended `<|im_end|>`, so the last end-of-turn zone is a forced
  ending.
- **Judge:** one Claude Sonnet 5.5 vote on whether the shown sample has the behaviour
  (`scripts/chive_judge.py`, `data/chive_judge.jsonl`). Its quotes become the citations.
  - CHIVE itself only gives the rate over 30 samples.
  - 294 of 298 judged samples show the behaviour; the judge refused 2.
- **Counterfactual experiments:** each rewrites whole turns. The page diffs them against the
  original turns (word level) and marks the edited tokens when you click an experiment. Hover a
  marked token to see what replaced it.
  - 103 of 1,944 experiments only add or move turns, so they have no span.
- **Agam's banks:** O-lens (s3d_rl600, 4 samples at T=1), multi-layer NLA, single-layer NLA-RL (L42)
  and J-lens (top 10 tokens, L0–62), at every token.
  - They're mirrored from his case viewer with
    `uv run python scripts/mirror_agam_readers.py splits/all300.txt data/banks/agam` (repo root, ~4
    GB).
  - The page fetches them 64 positions at a time (`/api/bank`). The "has our own read" underline
    ignores them.
  - A new reader (an SAE, say) is one more `READERS` entry with a `kind` the page knows how to draw.
- **Scored banks:** the O-lens bank samples get AR scores from `scripts/rescore_chive_banks.py`
  (deleted 2026-10-26, at commit `7296169`:
  https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/7296169/transcript-viewer/scripts/rescore_chive_banks.py)
  (`data/bank_scores/<pid>.jsonl`). The page uses them for bullet FVEs, the heatmap and the
  zone/punctuation tables wherever we have no greedy read of our own.
  - **Not run yet (2026-10-20), so the CHIVE bank samples show no FVE.** The script works, but two
    GPU launches produced no records. The first was blocked on the gated whiteners
    (`agu18dec/local-workspace`; request access on HF first). The second H100 pod loaded the server,
    then the driving agent stopped waiting before the test batch, and the pod idled out. That cost
    ~$6 in total. We then dropped it as not worth chasing.
  - Size: the relevant positions × 11 layers come to 244k items, an estimated 3–7 H100-hours
    (~$15–30). Every token would be ~13× that.
  - To run it, use `runpod/pod.sh` (Robbie's agents don't use SkyPilot). Do the `olens/CLAUDE.md`
    setup, start `pod/run.sh`, tunnel to it, then run the script with `--limit 3` to measure
    throughput before the full set.
- **Solve counts:** "T 3/5 · O 5/5" is how many of 5 draws in Agam's climb_test run (Opus 5) named a
  true cause, with the transcript alone vs with the O-lens (`data/viewer/cmp_climb_test.json`).

## Model organisms

`web/organisms.py` appends one pattern per LoRA model organism after the CHIVE incidents, under the
Organisms tab, from `data/runs/organisms/` at the repo root (HF dataset `runs/organisms/`; setup and
findings in `docs/model-organisms.md`). Each rollout is a transcript. The ones read with the lens
(sample 1 of each trigger and control prompt) have reads on the organism's activations and on base
Qwen's over the same tokens: the transcript header switches between them, and the other one shows
under each readout. Base Qwen's rollouts on the same prompt are at the bottom. The viewer can't make
new reads of these (the pod doesn't load LoRAs; `scripts/organism_olens.py` did, deleted 2026-10-26,
at commit `7296169`), and the current reads have no FVE scores.

## GLM 5.3

`web/glm.py` adds a GLM 5.3 tab: one pattern per env and arm of our GLM 5.3 evalcue runs
(`glm.ENVS`; now scheming_oversight_blk and hidden_goal_grade, the cfbench smoke envs), each rollout
read with Celeste's GLM-5.3 O-lens
([ceselder/glm-5.3-olens-counterfactual-investigator](https://huggingface.co/datasets/ceselder/glm-5.3-olens-counterfactual-investigator),
config `evalcue`).

- **Tokens:** Celeste's render through GLM-5.3's own chat template, cut with GLM-5.3's tokenizer, so
  her `pos` indexes them; messages are the spans between GLM's role tokens.
- **Reads:** her RL lens at layer 51, ~12 chosen punctuation positions per rollout (underlined):
  greedy under Decode greedy, the 4 T=1 samples under T=1 × 4. No FVE (her lens has no AR scores
  here) and no live reads (the pod serves Qwen's lens).
- **Labels:** the env's programmatic label (positive / negative); each transcript links to its
  Docent run.

## Thinking-on resamples

`web/thinking.py` adds one pattern per favourite (`docs/favorite-transcripts.md`) to the WeirdChat
tab, behavior `thinking-resample`: Qwen3.6-27B's reply to the same user turn, sampled again with
thinking on (T=1, top_p 0.95, top_k 20, up to 8192 new tokens). The reply's highlighted text starts
at its thinking. Two sources, both in `data/thinking.jsonl` (shared through `publish.py`):

- `s<k>`: 10 per favourite from our own bf16 vLLM (`scripts/thinking.py prep`, then `gen` on a
  RunPod pod), exact token ids.
- `or<k>`: 5 per favourite from `qwen/qwen3.6-27b` on OpenRouter
  (`scripts/thinking.py openrouter N`), whose providers may serve quantized weights; their text is
  re-templated into ids.

`scripts/thinking.py read PORT PARENT [TID ...]` reads one favourite's resamples against
`pod/server.py`. The prompt is shared up to `<think>\n`, so every sample's end-of-user-turn reads
are identical (greedy); samples only differ from the first thinking token on.

Code: `scripts/thinking.py` at commit `7296169`
(https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/7296169/transcript-viewer/scripts/thinking.py),
deleted 2026-10-26; the resamples and reads stay in the shared JSONL.

## Confession follow-ups

Neel Nanda's suggestion: take a weird transcript, add a user turn asking Qwen why it did the weird
thing, and read the lens on Qwen's answer and on the stop tokens after the question.
`web/followups.py` adds one pattern per original transcript to the WeirdChat tab, behavior
`confession-follow-up`, titled with its label:

- **favourite:** the three in `docs/favorite-transcripts.md`.
- **hospital, bad / medium / good:** six more rollouts of the hospital pattern (claims-called-911
  pg0039). Bad ones say "I am calling emergency services for you", medium ones say help is "on the
  way" without claiming to have called, and good ones say the AI can't call.

Each gets a neutral question ("Why did you respond that way?"). Those with something to name also
get a question naming the behavior ("Why did you say you were calling emergency services?"). Each
question gets a greedy answer and 4 answers at T=1. Greedy answers are read at layers 24–56 at their
end-of-turn zones, the question and the answer. Favourites, and their follow-ups, are read at every
token.

- **Tokens:** the prompt is the conversation plus the question, re-templated. The original reply
  therefore loses its empty `<think></think>` block, as in Qwen's multi-turn training format. So the
  follow-up's tokens differ from the original's from the reply on, and the two have separate reads.
- **Generating and reading:** `scripts/followups.py generate`, then `read`, against `pod/server.py`
  (it has a `/generate` endpoint) on 127.0.0.1:8802, e.g. a RunPod pod over an ssh tunnel.
  `read 8802,8812,...` spreads the reads over several servers, one thread each. It runs the viewer's
  own read code without a viewer, so the viewer's auto-precompute doesn't get in first. The answers
  go to `data/followups.jsonl`, which `publish.py` shares like the reads.
- **Code:** `scripts/followups.py` at commit `7296169`
  (https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/7296169/transcript-viewer/scripts/followups.py),
  deleted 2026-10-26; the follow-ups and reads stay in the shared JSONL.

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
- Each read records the `lens` it was made with; rows without one are the old `olens_s3d_rl600`. The
  key carries the lens (`web/server.py` `LENS`), so old reads never count as cached: the page shows
  them marked "old lens" and any read re-reads them. A lens backend started on an older commit (no
  `lens` in its reply) is refused.

## Sharing: the HF dataset and the public snapshot

[olens-viewer.pages.dev](https://olens-viewer.pages.dev) (its old name,
[weirdchat-olens.pages.dev](https://weirdchat-olens.pages.dev), 301s to it) is this page in
read-only mode (no GPU or queue controls), with links in the same `#p=…&t=…` form. Agam's CHIVE
banks are only published with `publish.py --banks`: they're his and unpublished, so ask before using
it. Its data lives in the project's HF dataset under `weirdchat/`, where each machine owns its own
files so publishes never clash:

- `weirdchat/reads/<machine>.jsonl`, `rollouts/<machine>.jsonl`, `stance/<machine>.jsonl`,
  `extras/<machine>.jsonl`: copies of this machine's `data/reads.jsonl`, `data/claude.jsonl`,
  `data/stance.jsonl` and `data/extras.jsonl` (NLA + J++). The machine name is `$WCO_MACHINE`, else
  the short hostname.
- `weirdchat/site/`: one gzipped JSON per transcript (the server's `/api/transcript`), per pattern,
  and the pattern list, plus NLA + J++ reads in 64-token chunks (`x/<tid>/<chunk>`), built from
  everyone's files. The page fetches these straight from HF (CORS is open).

```bash
.venv/bin/python scripts/publish.py          # push this machine's files, pull everyone's, rebuild + upload the site
.venv/bin/python scripts/publish.py --pull   # only pull other machines' files, to browse them in the local viewer
scripts/deploy_site.sh                       # (from the repo root) only when web/index.html changes
```

The local server reads the pulled files (`data/shared/weirdchat/`) alongside its own. The build
renders every transcript in forked workers (~15 s on a 14-core Mac) and uploads only the site files
that changed. That upload is the slow part when many files change: a full one (~27k files) takes
about 15 minutes, because the Hub commits ~1k files at a time. Writing to the dataset needs an HF
token with write access to it.

Publish only from code that matches `origin/main` (`publish.py` refuses otherwise; `--force`
overrides). The page is deployed from main, and the site JSON comes from your checkout's `web/`, so
publishing from older or unmerged code breaks the page and re-uploads every file.

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

## Case study page (`/case#t=<transcript id>`)

One transcript's user turn plus the first 20 reply tokens with every lens side by side: the O-lens
(11 layers), the [ceselder L42 NLA](https://huggingface.co/ceselder/qwen3.6-27b-nla-L42)
(`olens/src/olens/extras.py`; the RL LoRA is applied on top of the SFT LoRA, which its config names
as its base; FVE uses μ from 64 random WeirdChat transcripts, `pod/calib_ids.json`) and the
[J++ lens](https://huggingface.co/neuronpedia/jacobian-lens/tree/main/qwen3.6-27b/jpp) at all 63
layers. Readouts can be edited and re-scored. Vectors (`olens/src/olens/vectors.py`) come from J++
tokens, the O-lens or NLA AR of free text, or differences of these; they probe the transcript
(cosine per token) or steer 64 samples of its prompt (h ← h + α‖h‖v̂ at every position), judged with
WeirdChat's own judge prompt (`web/judge.py`, ported from
[transluceAI/weirdchat](https://github.com/transluceAI/weirdchat)) on Gemma 4 31B.

- NLA + J++ need ≥ 120 GB of GPU memory: `WCO_POD_YAML=sky/pod_h200.yaml` for Start GPU.
- `web/server.py --jpp-local` sends J++ reads to `web/run_lens.sh --extras jpp` on this machine.
- On a RunPod H200 instead (`runpod/pod.sh up` + `sync`, then `sky/pod_h200.yaml`'s setup by hand in
  `/workspace/repo/olens`): run `pod/server.py` there, tunnel it to 127.0.0.1:PORT, and
  `scripts/read_extras.py PORT [TID ...]` reads NLA + J++ at every token (default: the three in
  `docs/favorite-transcripts.md`) into `data/extras.jsonl`; `publish.py` then puts them on the
  public site. (`read_extras.py` was deleted 2026-10-26; at commit `7296169`:
  https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/7296169/transcript-viewer/scripts/read_extras.py.)
- Judge: `sky launch -c wco-judge -y -i 5 --down sky/judge_gemma.yaml`, tunnel 18812 → 8002, and
  `WCO_JUDGE_URL=http://127.0.0.1:18812/v1`. Agreement with WeirdChat's labels on the 64 stored
  rollouts of "Most-sober person…" prompt 1: 92%, κ 0.78. vLLM 0.25.1 needs transformers 5.10.0 for
  Gemma 4 (newer versions raise on its per-layer `head_dim`).
