# Transcript viewer

**Status (2026-10-26):** ACTIVE. Static page only: no server, no GPU, no read queue. The page reads
the HF dataset; new O-lens reads are batch jobs whose outputs land there. Every O-lens read on the
page (345k) is from the old `olens_s3d_rl600` lens and is marked "old lens"; re-reading with the
space lens was never scheduled ([docs/olens-space-migration.md](../docs/olens-space-migration.md)).

[olens-viewer.pages.dev](https://olens-viewer.pages.dev) browses the Qwen3.6-27B patterns of
[Transluce/WeirdChat](https://huggingface.co/datasets/Transluce/WeirdChat) (225 patterns, 25,792
transcripts), the 300 [CHIVE](https://huggingface.co/datasets/agu18dec/chive-forensics) incidents,
our model-organism and GLM 5.3 rollouts, token by token, with every agu18dec O-lens read we have
(`../olens`). Its old name, [weirdchat-olens.pages.dev](https://weirdchat-olens.pages.dev), 301s to
it; browsers keep the hash fragment, so old links work. Every view is a link (`syncHash`): the
pattern, transcript, token, layers, decode setting, FVE highlight and list filters all live in the
hash.

The page is `web/index.html`, one file, deployed to Cloudflare Pages by `scripts/deploy_site.sh`.
Its data is gzipped JSON under `weirdchat/site/` in the project's HF dataset
([Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens)),
built by `scripts/publish.py` from `web/build.py` and the data loaders next to it. Nothing runs
between the browser and HF.

## What the page shows

- Transcripts rendered through Qwen3.6-27B's chat template (`enable_thinking=False`); judge
  citations highlighted; end-of-turn zones (`<|im_end|>` through the next message's content) marked,
  since the lens reads best there.
- Click a token → its readouts at the chosen layers (20..60 step 4), where we have them (underlined
  tokens): bullets sorted by their own FVE, the joint FVE (NNLS refit of all bullets' AR directions
  in whitened space), NNLS weights. Greedy reads and T=1 × 4 samples are separate settings.
- FVE heatmap over the transcript (joint or top-bullet, any layer, per-transcript 5–95th percentile
  colour scale).
- Each read records the `lens` it was made with; rows without one are the old `olens_s3d_rl600` (the
  Chinese-prefix bug, `docs/olens-chinese.md`). The page shows them marked "old lens"; a
  current-lens read of the same token would replace them (none exist yet).
- AR quality grows with depth (held-out FVE 0.12 at L20 → 0.34 at L60), so FVE isn't comparable
  across layers.

## Tabs and data sources

**WeirdChat** (`web/build.py`): the 225 patterns, with WeirdChat's Elo tags, stance labels
(`data/stance.jsonl`) and Claude rollouts on the same prompts (`data/claude.jsonl`). Two kinds of
extra pattern sit in this tab:

- **Thinking-on resamples** (`web/thinking.py`, behavior `thinking-resample`): one pattern per
  favourite (`docs/favorite-transcripts.md`), Qwen3.6-27B's reply to the same user turn sampled
  again with thinking on (T=1, top_p 0.95, top_k 20, up to 8192 new tokens). `s<k>` samples came
  from our own bf16 vLLM (exact token ids), `or<k>` from `qwen/qwen3.6-27b` on OpenRouter (text,
  re-templated; providers may serve quantized weights). The reply's highlighted text starts at its
  thinking. The prompt is shared up to `<think>\n`, so every sample's end-of-user-turn reads are
  identical. Data: `data/thinking.jsonl`.
- **Confession follow-ups** (`web/followups.py`, behavior `confession-follow-up`): Neel Nanda's
  suggestion, a weird transcript plus a user turn asking Qwen why it did that, and Qwen's answer.
  Labels: the three favourites, and six more hospital rollouts (claims-called-911 pg0039: bad /
  medium / good). Each gets a neutral question ("Why did you respond that way?"); those with
  something to name also get one naming the behavior. Greedy answer plus 4 at T=1; greedy answers
  are read at layers 24–56 at the question, the answer and every end-of-turn zone, favourites at
  every token. The conversation is re-templated with the new turn, so the earlier reply loses its
  empty `<think></think>` block (as in Qwen's multi-turn training format) and the follow-up's tokens
  differ from the original's from the reply on. Data: `data/followups.jsonl`. Findings:
  `docs/confession-follow-ups.md`.

**CHIVE** (`web/chive.py`): one pattern per incident, numbered after WeirdChat's 225 so old `#p=`
links keep working.

- Rendered from CHIVE's own `token_ids` and per-token turn labels, not our chat template, because
  Agam's banks index into those ids.
- Only the final reply is Qwen's. Earlier assistant turns come from the source dataset (mostly
  WildChat, written by ChatGPT), and the page labels them.
- ✂ cut off: CHIVE sampled the final reply with a ~500-token cap. 176 of 300 hit it and stop
  mid-sentence, followed by an appended `<|im_end|>`, so the last end-of-turn zone is a forced
  ending.
- Judge: one Claude Sonnet 5.5 vote on whether the shown sample has the behaviour
  (`scripts/chive_judge.py`, `data/chive_judge.jsonl`); its quotes become the citations. 294 of 298
  judged samples show the behaviour; the judge refused 2. CHIVE itself only gives the rate over 30
  samples.
- Counterfactual experiments: each rewrites whole turns. The page diffs them against the original
  turns (word level) and marks the edited tokens when you click an experiment; hover a marked token
  to see what replaced it. 103 of 1,944 experiments only add or move turns, so they have no span.
- Agam's banks (O-lens s3d_rl600 4 samples at T=1, multi-layer NLA, NLA-RL L42, J-lens top-10 tokens
  L0–62, at every token) are mirrored from his case viewer with
  `uv run python scripts/mirror_agam_readers.py splits/all300.txt data/banks/agam` (repo root, ~4
  GB). They are his and unpublished, so the public site ships without them unless
  `publish.py --banks` (ask first). A new reader is one more `READERS` entry with a `kind` the page
  knows how to draw.
- Scored banks (`data/bank_scores/<pid>.jsonl`, AR scores for the O-lens bank samples) would give
  the CHIVE tab bullet FVEs and a heatmap. Not produced: `scripts/rescore_chive_banks.py` (deleted
  2026-10-26, at commit `7296169`:
  https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/7296169/transcript-viewer/scripts/rescore_chive_banks.py)
  worked but its two GPU launches produced no records (gated whiteners, then an idled-out pod), and
  we dropped it. The relevant positions × 11 layers are 244k items, an estimated 3–7 H100-hours.
- Solve counts ("T 3/5 · O 5/5"): of 5 draws in Agam's climb_test run (Opus 5), how many named a
  true cause, with the transcript alone vs with the O-lens (`data/viewer/cmp_climb_test.json`).

**Organisms** (`web/organisms.py`): one pattern per LoRA model organism, from `data/runs/organisms/`
at the repo root (HF dataset `runs/organisms/`; setup and findings in `docs/model-organisms.md`).
Each rollout is a transcript. The ones read with the lens (sample 1 of each trigger and control
prompt) have reads on the organism's activations and on base Qwen's over the same tokens: the
transcript header switches between them, and the other one shows under each readout. Base Qwen's
rollouts on the same prompt are at the bottom. These reads were made with the old lens and have no
FVE scores.

**GLM 5.3** (`web/glm.py`): one pattern per env and arm of our GLM 5.3 evalcue runs (`glm.ENVS`, the
cfbench smoke envs), each rollout read with Celeste's GLM-5.3 O-lens
([ceselder/glm-5.3-olens-counterfactual-investigator](https://huggingface.co/datasets/ceselder/glm-5.3-olens-counterfactual-investigator),
config `evalcue`). Tokens are her render through GLM-5.3's chat template, cut with GLM-5.3's
tokenizer, so her `pos` indexes them; messages are the spans between GLM's role tokens. Reads: her
RL lens at layer 51, ~12 chosen punctuation positions per rollout (underlined), greedy under Decode
greedy, the 4 T=1 samples under T=1 × 4, no FVE. Each transcript links to its Docent run.

**NLA + J++** (`web/extras.py`, `data/extras.jsonl`): for the three favourites, the
[ceselder L42 NLA](https://huggingface.co/ceselder/qwen3.6-27b-nla-L42) explanation (with cosine and
FVE) and the [J++ lens](https://huggingface.co/neuronpedia/jacobian-lens/tree/main/qwen3.6-27b/jpp)
top tokens at all 63 layers, shown under the O-lens readouts of a token. `scripts/read_extras.py`
made them (deleted 2026-10-26, at commit `7296169`).

## Getting new reads onto the page

Reads are batch jobs on RunPod: `scripts/olens_read_requests.py` at the repo root, run by
`runpod/olens_reads.sh` (`runpod/CLAUDE.md`). It reads every stop token of a transcript at the
chosen layers and scores the bullets with the AR, the same recipe as the viewer's reads (its score()
came from the deleted `pod/server.py`).

Its output format (`{id, token_index, layer, bullets: [{text, fve, coef}], joint_fve}` per line) is
not the viewer's
(`{key, tid, pos, layer, temperature, n, seed, samples: [{text, bullets, bfve, coefs, fve}], lens}`
in `data/reads.jsonl`), and its transcripts are rendered by vLLM's chat template from OpenAI-style
messages rather than from WeirdChat rows, so a converter (tid and token position mapping) is needed
before new batch reads show on the page. Nobody has needed one yet: the page shows the 345k reads
already in `weirdchat/reads/*.jsonl`. See `CODE_SMELL.md`. Batch reads show in expview instead, with
no converter ([docs/olens-reads-format.md](../docs/olens-reads-format.md)).

## Build and publish

```bash
cd transcript-viewer
uv venv .venv && VIRTUAL_ENV=.venv uv pip install transformers==5.16.1 tokenizers==0.23.1 pyarrow==25.0.1 jinja2 huggingface_hub
hf download Transluce/WeirdChat --repo-type dataset --local-dir data/hf \
  --include 'data/patterns.parquet' 'data/rubrics.parquet' 'data/prompts/qwen__qwen3.6-27b/*' 'data/transcripts/qwen__qwen3.6-27b/*'
.venv/bin/python scripts/publish.py --pull          # other machines' reads etc. into data/shared/weirdchat/
.venv/bin/python scripts/publish.py --preview /tmp/olens-site   # build the whole site locally, no HF writes
python -m http.server -d /tmp/olens-site            # http://localhost:8000/#pattern=36&transcript=…
.venv/bin/python scripts/publish.py                 # push this machine's files, pull everyone's, rebuild + upload the site
scripts/deploy_site.sh --branch preview             # (from the repo root) the page, to https://preview.olens-viewer.pages.dev
scripts/deploy_site.sh                              # the page, to production; only when web/index.html changes
```

Each machine owns its own files on HF, so publishes never clash:

- `weirdchat/reads/<machine>.jsonl`, `rollouts/`, `stance/`, `chive_judge/`, `followups/`,
  `thinking/`, `extras/`: copies of this machine's `data/*.jsonl` (`build.KINDS`). The machine name
  is `$WCO_MACHINE`, else the short hostname. `publish.py --pull` puts everyone else's under
  `data/shared/weirdchat/<kind>/`, which `web/build.py` reads alongside this machine's (last, so its
  records win).
- `weirdchat/site/`: one gzipped JSON per transcript (`t/<tid>`), per pattern (`pattern/<i>`), the
  pattern list and the overview, plus NLA + J++ reads in 64-token chunks (`x/<tid>/<chunk>`) and,
  with `--banks`, Agam's bank chunks. The page fetches these straight from HF (CORS is open), via
  the base URL in `index.html` (`window.WCO_STATIC` overrides it; `--preview` sets it to `./`).

The build renders every transcript in forked workers (~15 s on a 14-core Mac) and uploads only the
site files whose sha1 changed since the last upload (`weirdchat/site/manifest.json`). The upload is
the slow part when many files change: a full one (~27k files) takes about 15 minutes, because the
Hub commits ~1k files at a time. Writing to the dataset needs an HF token with write access to it.

Publish only from code that matches `origin/main` (`publish.py` refuses otherwise; `--force`
overrides, and code behind main is refused regardless). The page is deployed from main and the site
JSON comes from your checkout's `web/`, so publishing from older or unmerged code breaks the page
and re-uploads every file. Changing the JSON shapes (as the 2026-10-26 static-only change did,
dropping the queue fields) re-uploads every file once.

## History

Until 2026-10-26 this was a live viewer (`web/server.py` on bramble:8803) that read new tokens on a
SkyPilot GPU pod or a precompute fleet through a shared read queue, with a case-study page
(`web/case.html`: O-lens, NLA and J++ side by side, vector probes and steering judged by Gemma 4
31B). Robbie's 2026-10-26 decision: no viewer runs on a backend; every viewer is a static page over
the HF dataset, and reads are batch jobs. The deleted code (`web/server.py`, `web/pod.py`,
`web/readq.py`, `web/local.py`, `web/steer.py`, `web/judge.py`, `web/case.html`, `scripts/fleet.py`,
`pod/`, `sky/`) is at commit `b842346`:
https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/b842346/transcript-viewer.
