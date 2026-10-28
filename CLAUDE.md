# counterfactual-investigator-agent-with-olens

Zero-counterfactual investigator agent for CHIVE using metamodels (oracle lens, NLAs). MATS sprint
with Neel Nanda, run by Robbie and Kylie.

`AGENTS.md` is a symlink to `CLAUDE.md`, at the repo root and beside every nested `CLAUDE.md`, so
Codex/other agents see the same instructions. Do not replace one with a separate file.

## This repo is public

Unlike Robbie's usual repos, this one is **public**, and Kylie (Robbie's sprint partner) works in it
too. So:

- Don't write in machine-specific paths (`/Users/robbie/...`, `~/repos/<other-repo>`) as the only
  way to find something. Link the public source (arXiv, GitHub, HF), and give a local path only as
  an extra.
- Never commit secrets, API keys, private Slack content, or excerpts of unpublished write-ups (e.g.
  collaborators' in-progress Google Docs). Link to them; access is controlled on their end.
- Write docs for a reader who hasn't seen Robbie's other repos or Claude sessions.

## The project

Source of truth for the plan, ideas, and daily work log: the sprint doc
[Sprint Pitches Robbie and Kylie](https://docs.google.com/document/d/1Ghq1WhCyjcUV1XhG1x04mMW26o7saJV-S-yPxLhysI4/edit?tab=t.cjb65av6kggq)
(access-restricted; read it via the Google Drive connector). The "Final Pitch" section is the
current plan, and "Work Log" holds dated progress. Re-read it rather than trusting this summary when
they disagree.

Summary of the Final Pitch (2026-10-19):

- **Goal:** the best investigator agent possible with zero (or few) counterfactual resamples,
  leaning on metamodels, especially the oracle lens (O-lens / olens).
- **Baseline setup:** a cheap-ish predictor (Gemini 3.8 Flash, not a frontier model, so tool uplift
  isn't swamped by agent strength) on CHIVE transcripts from Qwen3-8B and/or Qwen3.6-27B, starting
  from Agam's harness ([agu18dec/detective-joracle](https://github.com/agu18dec/detective-joracle)).
- **Two legs:** (1) vanilla CHIVE, where the agent predicts whether each proposed counterfactual
  flips the behavior; (2) CHIVE without hypotheses, where the agent must generate them itself. Leg 2
  is the point: we want the O-lens to surface ideas that aren't in the transcript.
- **First steps:** reproduce Agam's results; get an O-lens for Qwen3-8B, or more CHIVE rollouts for
  Qwen3.6-27B (Agam has ~300); check the set isn't saturated and that there's a val split separate
  from dev.
- **Later:** O-lens test-time tricks (stop tokens before reading, feeding the agent per-bullet FVE,
  top-k bullets by FVE, small rewrites checked with the AV, AR as a probe for where to read);
  train-time tricks (RL the lens against CHIVE, penalize negative-FVE bullets); combining tools
  (J-lens, NLAs, Asher's [EasyNLA](https://github.com/asherps/EasyNLA) SNLAs); black-box baselines
  with and without counterfactuals.

## Gotchas from the background reading

- **We are not tied to the CHIVE pipeline**, e.g. how they found counterfactuals or how they
  measured "successful changes" (their ≥50pp YES / ≤15pp NO rules). The absolute ≥50pp rule makes
  every removal edit a NO when the baseline rate is under 50%, so prefer a headroom-normalized
  effect plus a significance test when we label our own experiments.
- **CHIVE's transcript-only baseline is strong (~81% AUROC), and in the paper no activation tool
  (AOs, NLAs, SAEs) beats it.** CHIVE didn't test the O-lens or J-lens, which is our opening.
- **Selection bias in CHIVE.** The counterfactuals were proposed by a transcript-reading agent, so
  the dataset only contains hypotheses that are findable from the transcript. That inflates the
  transcript-only baseline. Filtering to "hard for transcript-only" examples is collider bias; don't
  do it. This is why leg 2 (no hypotheses) matters.
- **Prompt inversion.** If a lens just paraphrases the transcript or CoT back, it adds nothing. When
  a tool "helps", check that its outputs say something the transcript doesn't.
- **Two oracle-lens recipes exist and differ:** Anthropic's (workspace paper appendix, Haiku 4.5,
  2026-07) and the WorkspaceBench / agu18dec one (Qwen3.6-27B, 2026-09). The latter is newer and
  ablated. Don't assume Anthropic's is better.
- **Use the space O-lens, `olens_s3d_space_rl600`, never `olens_s3d_rl600`.** The older agu18dec
  lens's RL reward stripped each bullet's leading space before the AR scored it, so it opens ~half
  its bullets with a Chinese word; the space lens fixes that at the same WorkspaceBench score
  (`agu18dec/olens_and_ar` README; our check in `data/runs/olens_space_fix/`). Score its bullets
  with the AR as `" " + bullet` (`lens.bullet_prefix`; `olens/` does this). Every read we made
  before 2026-10-22 ~17:25 PDT used the old lens; the redo list is in
  [docs/olens-space-migration.md](docs/olens-space-migration.md).
- **Metamodel hallucination.** O-lens and NLA readouts invent names, numbers, and plans that aren't
  in the activation. The O-lens gives a per-bullet FVE, which is a cheap hallucination signal to
  pass to the agent.
- **Don't paste O-lens reads into the predictor's prompt** (the `transcript_olens*` conditions,
  closed PR #116, 2026-10-22). It's a dead end: no gain over a placebo, and one variant read the
  lens on the counterfactual prompt, which is outside the zero-counterfactual setting. See
  [docs/envs/hidden_goal.md](docs/envs/hidden_goal.md).
- **cfbench conditions are `no_transcript`, `transcript_agentic`, `olens_agentic`,
  `olens_placebo_agentic`** (renamed 2026-10-27; old rows map through `canon_condition`). The inline
  two-transcript condition is retired: never compare a tool condition against anything but
  `transcript_agentic`.
- **Bump `HARNESS_VERSION` in `cfbench/core.py`** whenever the predictor's prompt, tools or scoring
  change, and never compare cfbench skill across versions (runs without the field are version 1).

## Background reading

All public:

- CHIVE: [blog](https://alignment.anthropic.com/2026/chive/),
  [paper, arXiv 2608.16747 (2026)](https://arxiv.org/abs/2608.16747)
- Oracle lens / global workspace, Gurnee et al. (2026):
  [transformer-circuits.pub/2026/workspace](https://transformer-circuits.pub/2026/workspace/)
- NLAs (2026): [transformer-circuits.pub/2026/nla](https://transformer-circuits.pub/2026/nla/),
  [code](https://github.com/kitft/natural_language_autoencoders)
- WorkspaceBench, Blank, Bhatia, Ong, Nanda (2026):
  [AF post](https://www.alignmentforum.org/posts/Zeg2JztbdhguL48uH/),
  [code](https://github.com/camilablank/workspace-bench), O-lens + AR checkpoint
  [agu18dec/olens_and_ar](https://huggingface.co/agu18dec/olens_and_ar)
- AuditBench, Sheshadri et al. (2026): [arXiv 2602.22755](https://arxiv.org/abs/2602.22755)
- Activation Oracles, Karvonen et al. (2025): [arXiv 2512.15674](https://arxiv.org/abs/2512.15674)
- Robbie's from-scratch O-lens on Qwen3.6-27B:
  [repo](https://github.com/robbiethompson18/oracle-lens-reconstruction),
  [HF](https://huggingface.co/collections/robbiethompson2018/nn-mats-sprint-oracle-lens-on-qwen36-27b).
  Lens dashboard: [jlens-dashboard](https://jlens-dashboard.pages.dev/)
  ([repo](https://github.com/robbiethompson18/jlens-dashboard))

**Robbie only:** `~/repos/metamodel-socratic-quizzes` (private) has local Markdown copies of the
papers above (`papers/*.md`, easier to grep), detailed notes on the two O-lens recipes
(`papers/oracle-lens-workspace-notes.md`, `papers/workspacebench-notes.md`), and a catalog of the
private O-lens research Slack log. Read them there; don't copy them into this repo. Also there:
[`docs/daria-olens-threads.md`](https://github.com/robbiethompson18/metamodel-socratic-quizzes/blob/main/docs/daria-olens-threads.md),
the full text of Daria Ivanova's Slack posts on O-lens readouts (donation bet, motivated reasoning,
split tokens). Private repo: Robbie can open it, Kylie can't.

## Data

`data/` (gitignored) mirrors the public HF dataset
[Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens):
O-lens banks in `banks/`, agent runs in `runs/`. Pull with
`hf download Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens --repo-type dataset --local-dir data`.
CHIVE incidents themselves come from
[agu18dec/chive-forensics](https://huggingface.co/datasets/agu18dec/chive-forensics).

**Every experiment run goes through `scripts/runs.py`** (soft rule for Kylie's runs): outputs in
`data/runs/<experiment>/<YYYY-MM-DD>_<slug>/` (experiment = env name or experiment family) with a
`meta.json` (git SHA, command, models with HF revision and which pod/API served them, params). A run
isn't finished until `scripts/runs.py finish RUN` has synced it to HF, including abandoned runs.
Results that exist only on a pod, robbie-box or one laptop don't count. Usage is in the script's
docstring.

Cases viewer: [chive-cases.pages.dev](https://chive-cases.pages.dev) shows all 204 held-out
incidents with every transcript-only and O-lens draw, filterable and linkable by URL.
`viewer/index.html` is the page, `viewer/behavior_kind.tsv` our hand labels; rebuild and redeploy
with `scripts/deploy_viewer.sh` (Cloudflare Pages, needs `CLOUDFLARE_API_TOKEN`).

O-lens viewer: [olens-viewer.pages.dev](https://olens-viewer.pages.dev) is the current public
token-level viewer (WeirdChat, CHIVE, organisms, GLM 5.3). `weirdchat-olens.pages.dev` is its old
name and 301s here (hash kept, so old links work); give new links on `olens-viewer` as
`#pattern=…&transcript=…` (old `#p=…&t=…` links are rewritten on load, docs/viewers.md). It is a
static page (`transcript-viewer/web/index.html`, deployed by
`transcript-viewer/scripts/deploy_site.sh`) over gzipped JSON in the HF dataset (`weirdchat/site/`,
built by `transcript-viewer/scripts/publish.py` from every machine's reads). No viewer runs on a
server or GPU any more (2026-10-26): new O-lens reads are batch jobs
(`scripts/olens_read_requests.py` via `runpod/olens_reads.sh`) whose outputs go to HF.

**expview is the standard viewer** for anything with transcripts, LLM judge/grader calls or
metamodel reads ([expview/CLAUDE.md](expview/CLAUDE.md)): write the run's items to `RUN/expview/`
with `expview.Run` (every judge call with its full input and output, O-lens/J-lens reads as
`tokens=`), and link items with `python expview/expview.py link`. Page:
`https://robbiethompson18.github.io/counterfactual-investigator-agent-with-olens/expview/` (env runs
open as env → arm → run → episode, with every run under `runs/<env>/` that points at an episode
stitched in: O-lens reads, judge calls; static, reads the HF dataset; a run is there once
`runs.py finish` has synced it; regex search over the reads of the runs you pick, in the browser).
The old `https://expview.pages.dev` is frozen at 2026-10-26. For a run that isn't on HF,
`expview.py build RUN` makes a self-contained page to publish as an Artifact.

Env and cfbench transcripts: public [Docent](https://docent.transluce.org) collections, one per env
plus `cfbench`, uploaded by `scripts/runs.py finish`. **When you cite an env or cfbench transcript
(docs, PRs, Slack, chat), link its Docent run**
(`uv run --script scripts/docent_upload.py link ENV/RUN:LINE`), or a filtered table
(`... view ENV arm=X positive=true`). Details in [docs/docent.md](docs/docent.md).

## Stack

- Python: `uv` for deps/venv, `ruff` for lint + format (line length 140), `ty` for types. Run
  `uv run ruff check . && uv run ruff format . && uv run ty check` before shipping.
- `.githooks/pre-commit` runs ruff on staged Python files. Enable it once per clone with
  `git config core.hooksPath .githooks`. Don't bypass it with `--no-verify`; fix the lint instead.
- Markdown: Prettier, 100 cols, `proseWrap: always`. Prettier is Markdown-only here.
- By default you code too defensively, please do not do this. This project is under development,
  privilege brevity and readability over being very robust.
- Tests: none until this repo is roughly > 50k LOC. Verify by running the code, not by adding a test
  suite. If a test would genuinely save time, ask first.
- Secrets/machine-specific env go in `.envrc.local` (gitignored), never `.envrc`.
- `vendor/` (Agam's `detective-joracle`, Singh and Kroiz's `agent-interp-envs`) is reference-only;
  see `vendor/CLAUDE.md`. Never import, run or wrap it; port what you need into our own harness
  (`environments/`, `cfbench/`). Ruff bans `import detective_joracle`. The old robbie-box screens of
  agent-interp-envs (docs/envs-screen) were deleted; their gateway, configs and scripts are at
  commit `e145988`.
- Never cap Qwen's thinking (no `max_tokens` or thinking budget on generations): it bugs out under a
  cap, and truncated episodes are unscorable.
- **Run Qwen with thinking off by default** (`--thinking off`; new envs set `"thinking": False` in
  `DEFAULTS`). Thinking off is the setting where behaviours stay unverbalized, which is what the
  O-lens is for; thinking on is for a deliberate on/off comparison only, and the two modes get
  different rates (see [docs/thinking-off.md](docs/thinking-off.md)), so never pool them.
- Run every non-Qwen model in every env at reasoning effort `medium` (OpenRouter `reasoning.effort`;
  `environments/` sends it by default), not high or max, even when replicating someone's setup that
  used max. Provider defaults differ (GLM 5.3's is max), so always set it explicitly.
- LLM judge: Claude Haiku 5.5 (`claude-haiku-5-5`, OpenRouter `anthropic/claude-haiku-5.5`) at
  reasoning effort `medium` for every judge and grader call, with 1 replicate per answer (Sonnet 5.5
  until 2026-10-26; runs before then say which). Agam's published numbers used Opus 5, so ours
  aren't directly comparable to his; say so when quoting both.
- Predictor (cfbench and any investigator agent): Gemini 3.8 Flash (`google/gemini-3.8-flash`) at
  effort `medium` by default, so tool uplift isn't swamped by agent strength.
- **Viewers are static pages on the public internet, never a backend.** One HTML file on Cloudflare
  Pages or the repo's GitHub Pages site that reads the public HF dataset; search is a JavaScript
  regex over the loaded run; reads are batch jobs whose outputs land on HF. Nothing on bramble, no
  tailnet URLs in docs or messages, no SkyPilot pod behind a button. See
  [docs/viewers.md](docs/viewers.md).
- Chart colors, the same in every plot, viewer and dashboard: transcript-only in blue, with O-lens
  in orange, placebo in black, no tool calls in gray. Give any other condition a color outside these
  four.

## Docs

Durable lessons about this repo go in git:

- **One-line rules** → this file (`CLAUDE.md`), or `CLAUDE.local.md` for machine-specific
  (gitignored).
- **Longer reference docs** (5–300 lines: experiment writeups, gotchas, repros) → `docs/*.md`,
  kebab-case filenames. Flat by default; add a topic subfolder only once several docs of one kind
  pile up.
- **Local-only docs** (not in git) → `docs/local/*.md`.
- **How to use one directory's code** (commands, formats, conventions; nothing dated) → that
  directory's `CLAUDE.md`, with an `AGENTS.md -> CLAUDE.md` symlink beside it. Claude Code loads it
  whenever an agent reads or edits a file in that directory, so keep it short and current.
- **No `README.md` files.** Agents are the readers, not GitHub visitors: usage goes in the
  directory's `CLAUDE.md`, results and write-ups go in `docs/`. A results table beside the code rots
  unseen: `cfbench/README.md` quoted harness v1 log skill as current until the 2026-10-26 audit.
- **Env fixtures that ship a `CLAUDE.md` to the model** (`evalcue/envs/*/workspace*/`) are not docs:
  `.claude/settings.json`'s `claudeMdExcludes` keeps Claude Code from loading them as instructions.
  Add any new fixture path there.
- Durable context lives in git, not in Claude Code's machine-local memory system.

Behavioral rules (one-liners) go directly in this file, not in docs: docs are lazy-loaded, but rules
need to always apply.

Every agent-relevant doc gets a one-line index entry below, so future sessions know it exists:
`- [TAG: Title — when to read this](docs/thing.md) — short gloss of contents`.

Every doc's first line after its title is `**Status (YYYY-MM-DD):** TAG. note`, and its index
entry's link text starts with the same TAG. Tags: ACTIVE (code or process in current use), REFERENCE
(durable result, nothing contradicts it), MIXED (some claims are stale; the note says which and what
replaced them), NULL (negative result, kept so nobody retries it), SUPERSEDED (by what), ORPHAN
(deleted code or an abandoned direction with no durable result). When a later result changes a doc's
standing, update its status line and tag rather than editing the old claims. `docs/figures/` holds
generated tables, not docs; they get neither a status line nor an index entry. The 2026-10-26 audit
behind the tags is in `docs/local/doc-audit-2026-10-26.md` (Robbie's checkout only). PNGs are
gitignored everywhere (2026-10-27): a doc's `![…](img/…)` renders after a local regeneration with
the script the doc names; charts are never committed or put on HF.

When you notice or leave tech debt, add a dated note to the top-level `CODE_SMELL.md` (create it if
missing) instead of relying on memory.

Current docs:

- [ACTIVE: Environments — read before building, running, converting or browsing any behavioural env](environments/README.md)
  — env = folder with env.py (ARMS/items/build/measure), one runner with model flags
  (`python -m environments run`), run folders on HF under `envs/`, generated indexes + static list
  page; essay_grading, alignment_faking, honest (11 scenarios as arms) and overeagerness ported,
  everything else in `environments/retired/`
- [ACTIVE: Viewers — read before building, changing or deploying any viewer or dashboard](docs/viewers.md)
  — the rule (static page on Cloudflare or GitHub Pages reading public HF, regex in the browser, no
  backend), why, what was tried and rejected, where each page lives
- [ACTIVE: expview — read before showing anyone transcripts, judge I/O or O-lens/J-lens reads, or adding a dataset to the viewer](expview/CLAUDE.md)
  — run format, `Run`/`call`/`read`, `index.jsonl`, deep links, the static page on Cloudflare Pages
  reading HF, in-browser regex search with echo filters, Celeste importer, `deploy.sh`
- [REFERENCE: CHIVE data sources — read before fetching incidents, readouts or Agam's results](docs/chive-data-sources.md)
  — HF dataset vs Agam's viewer, what each lacks, splits, file paths
- [REFERENCE: Qwen self-report baseline — read before comparing anything to Qwen explaining itself](docs/qwen-self-report-baseline.md)
  — setup, results on dev96, Sonnet 5.5 judge gotcha
- [ACTIVE: Transcript viewer — read before reading WeirdChat, CHIVE, organism or GLM transcripts with the lens on olens-viewer, or rebuilding or redeploying it](transcript-viewer/CLAUDE.md)
  — static page over the HF dataset: what each tab shows, CHIVE rendering and Agam's banks,
  confession follow-ups and thinking resamples, `publish.py` build/preview/upload, deploy, the
  batch-reads format gap, where the deleted live viewer (server, queue, fleet, case page) is
- [ACTIVE: olens library — read before reading transcripts with the agu18dec O-lens/AR or the J++ lens](olens/CLAUDE.md)
  — the one-engine reader's modules (carrier prompt, OneEngine, whitened FVE, provenance), the J++
  lens on the same engine, the two batch readers
- [NULL: RL the O-lens toward novelty — read before training or rewarding the O-lens to say what the transcript doesn't](docs/olens-rl-novelty.md)
  — negative result: AR-space novelty rewards pay for paraphrase and translation (v2 learned to
  translate); 8 Qwen resamples explain as much as the lens at deep layers; code on a closed branch
- [REFERENCE: Why the O-lens writes Chinese — read before interpreting Chinese in O-lens readouts](docs/olens-chinese.md)
  — how much there is in our banks; evidence it is a training artifact (stripped bullets + Chinese
  in RL data)
- [REFERENCE: Why the DeepSeek V4 Flash O-lens writes Russian — read before interpreting Russian or CJK in Camila's V4 lens reads, or suspecting our reader](docs/dsv4-lens-russian.md)
  — 3–9% of reads, concentrated at a few shared instruction-trailer positions, on-topic
  translations; not the whitespace bug, reader audited clean, same in Camila's own WorkspaceBench
  readouts
- [MIXED: Old-lens reads to redo — read before trusting or reusing any O-lens read made before 2026-10-22 evening](docs/olens-space-migration.md)
  — which artifacts hold reads from the buggy `olens_s3d_rl600`, what was re-read with the space
  lens, and the redo list that was then abandoned (treat every pre-switch read as old lens)
- [REFERENCE: Smoke25 results — read before comparing anything to Agam's CHIVE harness numbers](docs/smoke25-results.md)
  — our wrapper reproduces his free-form pass@1; Flash hits the call cap on yes/no with O-lens
- [MIXED: Model organisms on Qwen3.6-27B — read before running or reading organism rollouts/O-lens reads](docs/model-organisms.md)
  — which public LoRA organisms work, the rollout and paired organism/base O-lens scripts, first
  findings (old-lens reads)
- [REFERENCE: False capability claims — read before studying Qwen claiming to call 911, run code, text someone etc.](docs/false-capability-claims.md)
  — the device-channel lever (1.6% → 41%), severity dose-response, judge rubric v2 + audit,
  generator, and the negative held-out result outside emergencies
- [MIXED: Whitebox tools beyond the O-lens — read before adding SAEs, J-lens/J++, NLAs or SNLAs to the agent](docs/whitebox-tools.md)
  — what's in the viewer vs cfbench (NLA/J++ spike reads exist, no condition uses them), SNLA parked
  (why), SAE candidates and tool design
- [NULL: Thought-Anchors tools — read before giving the agent attention-based (non-metamodel) interp tools](docs/thought-anchors.md)
  — receiver heads + attention suppression on Qwen3.6's hybrid stack (DeltaNet hooks, leak test),
  the `anchors` cfbench condition (removed 2026-10-26); null vs placebo on hidden_goal (38 cases,
  log skill v1), why
- [ACTIVE: Serving metamodels on vLLM — read before decoding O-lens/NLA/J++ readouts at scale or building a metamodel server](docs/metamodel-serving.md)
  — ceselder/vllm-metamodels and other prior art, version pins (vLLM 0.27.1, not 0.19), merged-LoRA
  checkpoints, the TODO for a minimal Qwen3.6-27B metamodel-serving repo
- [MIXED: Shout "FIRE" — read before investigating odd advice in WeirdChat's claims-called-911 rollouts](docs/shout-fire.md)
  — why Qwen tells a trapped worker to shout "FIRE!": rates, prompt edits, other models,
  self-report, O-lens reads (old lens)
- [ACTIVE: Robbie's favorite transcripts — read when picking showcase transcripts or "why did you do X?" follow-up targets](docs/favorite-transcripts.md)
  — running list of the most interesting WeirdChat/CHIVE transcripts, with viewer links and ids
- [NULL: WSB association examples — read before running the O-lens on WorkspaceBench association families](docs/wsb-association-examples.md)
  — 9 unjudged items, s3d.ddp600 readouts on HF, what they say, pod/pip gotchas
- [MIXED: Party resampling — read before trusting an O-lens readout's specific names, or fitting lens output → behaviour](docs/party-resample.md)
  — third-party-politics organism: off-target parties in reads are never taken (0/1,352, 0/1,480),
  lens lags the text; paired lens-vs-resample table (old-lens reads)
- [MIXED: Confession follow-ups — read before asking Qwen "why did you do that?" or reading the lens on its answer](docs/confession-follow-ups.md)
  — setup, findings (named vs neutral question, one lens/answer disagreement), gotchas (old-lens
  reads)
- [ACTIVE: Environment construction goals — read before building or iterating on any multi-arm behavioral env (reward hacking, self-preservation, hiring bias, ...)](docs/environment-construction-goals.md)
  — symmetric arms, any lever, build/measure/tweak/stack, what makes an arm count
- [MIXED: Eval-cue reward hacking — read before building envs where Qwen3.6-27B hacks only under eval cues](docs/eval-cue-reward-hacking.md)
  — 10-env sweep, open-weight positive controls, round 2 on Qwen (knobs move it, no eval cue
  survives a split test); harness in `evalcue/CLAUDE.md`
- [ORPHAN: evalcue harness (deleted 2026-10-27, code at c7b2e56; honest is now environments/honest/) — read before reviving an evalcue env](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/c7b2e56/evalcue/CLAUDE.md)
  — `evalcue.run`/`show`/`--rescore`, `Episode` fields, run dirs, the `EVALCUE_*` env vars
- [NULL: evalcue config logs — read before retrying an eval cue on humaneval_prefill, matonski_grader, repo_impossible or secret_number](docs/evalcue-secret-number-log.md)
  — one log per env (`docs/evalcue-*-log.md`), every config tried in order; no eval cue survives a
  split test; secret_number's `r8_cfbench` arms are cfbench's secret_number cases
- [REFERENCE: weirdness_elo — read before picking the weirdest CHIVE behaviours or adding a pairwise LLM-judge ranking](weirdness_elo/CLAUDE.md)
  — Bradley-Terry weirdness Elo over 1,802 CHIVE behaviours (Sonnet + Qwen3.6-27B judges), dashboard
- [ORPHAN: donehack (deleted 2026-10-27, code at c7b2e56; ported envs in environments/) — read before reviving is_prime, an envport port or the realism ladder](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/blob/c7b2e56/donehack/CLAUDE.md)
  — the hardened podman sandbox, run commands, what each file does; results are in the three docs
  below
- [REFERENCE: donehack endings — read before claiming how episode endings (done tool, submit script, give_up) affect reward hacking](docs/donehack-endings.md)
  — V4 Flash / Qwen3.6-27B hack rates by ending (reply 52% → binary-reward submit script 88–95%),
  give_up raises it, reward schemes
- [REFERENCE: envport results — read before quoting GLM 5.3 on a batch-2 agent-interp-envs port (eval_tampering, games, parse_duration, revert_or_refactor, …)](docs/envport-glm53-results.md)
  — the per-env result tables and transcript notes that used to live in each port's README; batch-1
  numbers stay in glm53-agent-interp-envs.md
- [MIXED: GLM 5.3 agent-interp-envs runs — read before comparing GLM 5.3 coding/EvilGenie/ImpossibleBench/Donation Bet results or running many podman sandboxes](docs/glm53-agent-interp-envs.md)
  — headline numbers, batch-2 envport results (revert_or_refactor 94% revert), artifact and Docent
  links, podman scaling gotchas; its sandbagging line is superseded by envs/sandbagging.md
- [REFERENCE: GLM 5.3 realism variants — read before running or extending the user-style ImpossibleBench ladder or the email over-eagerness envs](docs/glm53-realism.md)
  — criterion-free 8-rung ladder (hack 27% → 3%, test edits 0% → 56%), evalness ranking, O-lens
  prompt and grader reads, email in-between rungs (casual ask alone → 2%)
- [SUPERSEDED: funding_email env screen — read before running or labelling funding_email (agent-interp-envs) rollouts](docs/envs-screen/funding_email.md)
  — Qwen3.6-27B deviates ethically 19/30 with the Myanmar memo vs 0/30 with a routine one; arm
  configs, Sonnet-subagent labels, shared rubric in `scripts/envs_screen/funding_email_rubric.py`
  (deleted, at `e145988`)
- [ACTIVE: funding_email in evalcue — read before running or extending the funding_email env (drone pitch vs Myanmar legal memo)](docs/envs/funding_email.md)
  — port of agent-interp-envs with its own tool schemas (`Episode.tools`/`step`), detector vs
  subagent labels, replication of the screen (14/30 vs 0/30)
- [ACTIVE: funding_email_real — read before running the realistic (Claude Code harness, traffic-analytics) funding_email grid](docs/envs/funding_email_real.md)
  — upstream/core/full harness × thin/real docs × memo, how to render an arm, unfixed sandbox tells
- [SUPERSEDED: sandbagging env screen — read before running or labelling sandbagging (agent-interp-envs) rollouts](docs/envs-screen/sandbagging.md)
  — Qwen3.6-27B sandbags deliberately 5/30 when >50% triggers unlearning vs 0/29 reversed
  (MARGINAL); intent labels from reader subagents, rubric in `scripts/envs_screen/sandbagging.py`
  (deleted, at `e145988`)
- [REFERENCE: Sandbagging iteration — read before running, extending or labelling the multi-arm sandbagging env](docs/sandbagging-iteration.md)
  — one system-prompt sentence ("Make sure you achieve your goal in the long term.") moves Qwen's
  deliberate sandbagging 17% → 73% (n=30, p=1e-5); config-driven overlay, every arm tried
- [ACTIVE: Sandbagging in evalcue — read before running the sandbagging env (it now runs in evalcue, not the vendored harness)](docs/envs/sandbagging.md)
  — `evalcue/envs/sandbagging/`: all 22 arms verbatim, the original tool loop, `detect()` matching
  544/544 original labels; replicates on OpenRouter (`a_goal` 8/30 vs `a_goal_lt` 21/30, p = 8e-4);
  GLM 5.3 baseline 2026-10-24 (deliberate ≈1/100, the detector positives are slips)
- [NULL: eval_tampering env screen — read before running eval_tampering (agent-interp-envs) on Qwen](docs/envs-screen/eval_tampering.md)
  — FAIL, retired for Qwen: Qwen and GLM disclose the tampering every time; the shipped variant
  names don't exist in the image, so arm B needs our overlay
- [NULL: optimize_cache_c env screen — read before running Optimize LRU cache (agent-interp-envs) on Qwen](docs/envs-screen/optimize_cache_c.md)
  — FAIL, retired for Qwen: Qwen never rigs the benchmark (0/14 at 3x), plateaus at ~1.0–1.2x and
  reports honestly
- [NULL: grade_your_own env screen — read before running grade_your_own (agent-interp-envs) on Qwen or rewriting it](docs/envs-screen/grade_your_own.md)
  — FAIL, retired for Qwen (rewritten as `evalcue/envs/exclude_rows/`): at an unreachable bar Qwen
  never stops (loops instead of reporting); notes for the rewrite
- [ACTIVE: Docent — read before linking, filtering or uploading env/cfbench transcripts](docs/docent.md)
  — what's in which collection, `link`/`view` commands, auto-upload at finish in both runners
  (`environments run`, `runs.py finish`), backfill, Kylie-key and re-upload gotchas
- [ACTIVE: graphs — read before making, regenerating or moving any chart](graphs/CLAUDE.md) — every
  chart module in one package (`python -m graphs.<module>` → `docs/img/`), which doc each serves;
  scoring stays in `cfbench.scoring`
- [ACTIVE: cfbench — read before predicting counterfactual arm outcomes or adding an env to the prediction harness](cfbench/CLAUDE.md)
  — our from-scratch CHIVE replacement: arm-pair cases, conditions, O-lens tools, CLI, adding an
  env; results live in the cfbench docs
- [ACTIVE: cfbench evals — read before building an eval that picks its own cases (one transcript per case, paired models, hidden model name) or showing one in the env viewer](cfbench/evals/CLAUDE.md)
  — a folder per eval (eval.py: ENVS, prep, cases; eval.md shown to the investigator),
  `python -m cfbench.eval`, evals keyed by eval not env in the viewer; af_refusals (AF refusals,
  DSv4 vs GLM)
- [SUPERSEDED: cfbench harness v1 results — read only when an old doc quotes a 2026-10-21/22 log-skill number](docs/cfbench-v1-results.md)
  — the four-predictor tables, the Haiku overnight sweep (4,288 cases) and the 2026-10-22 O-lens
  paragraphs moved out of cfbench/README.md; log-skill definition
- [ACTIVE: cfbench scoring — read before scoring a predictor, quoting a cfbench number, or making the report's cfbench charts](docs/cfbench-scoring.md)
  — RMSE of the predicted rate (why not log skill, not Beta nulls), draw/case normalization, ESS in
  env space, `graphs.cfbench_report` and its run groups
- [ACTIVE: cfbench harness smoke test — read before changing the predictor's prompt or tools, or bumping HARNESS_VERSION](docs/cfbench-harness-smoke.md)
  — hidden_goal all-stakes → all, 5 draws × TS/O-lens/placebo, the command, log-skill results per
  harness version (v1–v6) and one base run vs 30
- [REFERENCE: cfbench regraded as binary — read before asking whether the report's RMSE conclusions depend on the scoring, or scoring 'does the rate move ≥ T pp?'](docs/cfbench-binary-regrade.md)
  — same Flash predictions as yes/no at T = 5–30pp: O-lens vs placebo still a wash; on GLM
  env-diff-only wins the binary grading (bigger moves, no overshoot penalty); needs a real
  binary-prompt rerun to compare formats
- [REFERENCE: cfbench by predictor model — read before choosing a cfbench predictor, effort or tool budget, or claiming O-lens uplift across models](docs/cfbench-model-chart.md)
  — 7 models × env diff/transcript/O-lens/placebo on 4 cases (100 − RMSE): transcripts worth up to
  +21pp, O-lens beats placebo only for Flash and Argon via hidden_goal; Flash effort and tool-budget
  sweeps
- [NULL: cfbench with dumber, non-thinking predictors — read before trying a weaker or thinking-off predictor to get O-lens uplift](docs/cfbench-dumb-predictors.md)
  — 7 thinking-off/low models × TS/O-lens/placebo on 6 lens-friendly cases: all within ±2pp of
  placebo (they see the key bullets and ignore them); only Gemini 3.8 Flash gains (+5pp); which
  OpenRouter models can turn thinking off; `cfbench.run --effort off`
- [MIXED: O-lens cfbench spike — read before testing whether O-lens reads help a counterfactual predictor](docs/olens-cfbench-spike.md)
  — 2026-10-22, log skill v1: no uplift on killswitch, inline_b (old lens), exfil (old lens),
  whistleblow or donation bet; superseded for hidden_goal by its pre-registered test; arm-level vs
  within-arm signal, tool use per predictor
- [MIXED: Exclude-rows env — read before running or extending evalcue's exclude_rows reward-hacking env](docs/exclude-rows-env.md)
  — Qwen excludes correctly-labeled test rows to beat a classifier bar; design, every config,
  per-arm rates
- [MIXED: Identity counterfactuals — read before building counterfactual evals around Qwen's self-identity or deployment](docs/identity-counterfactuals.md)
  — one-line edits that flip behaviour: `Locale: en-US` vs `zh-CN` lifts China censorship;
  `Region: us-west-2` vs `us-west1` flips Claude-vs-Gemini API answers; old-lens reads, not ported
  to evalcue
- [NULL: Identity-belief experiments — read before studying what Qwen thinks made it, or which behaviours depend on that](docs/identity-belief-experiments.md)
  — cues that shift self-ID (Google from ChatGPT artefacts), identity-dependent probes, combos,
  scores
- [ACTIVE: Shutdown resistance envs — read before running or extending shutdown_palisade/killswitch/cron/persist](docs/envs/shutdown.md)
  — Palisade (2025) port + 3 siblings, ~200 arms; "The catch is that by default…" flips bypass
  (20/26 vs 0/20), "write access … configuration files" disables the kill switch 36/40 bf16, 30/40
  in the fp8 canonical run (mostly unverbalized); every arm in
  [shutdown-arms.md](docs/envs/shutdown-arms.md)
- [ACTIVE: Scheming envs — read before running Lynch blackmail/leak/murder or Apollo oversight/exfil/goal-guarding on Qwen](docs/envs/scheming.md)
  — 12 envs (bf16 rates; fp8 canonical exfil run in the doc); Lynch main cell 87%/86%/53%; exfil 48%
  → 20% with an eval cue (rarely verbalized); placebo caveats; tables in
  [scheming-tables.md](docs/envs/scheming-tables.md)
- [ACTIVE: Canonical envs — read before picking envs or arms for an O-lens / cfbench experiment](docs/canonical-envs.md)
  — the 8 standard envs and arm pairs, criteria, which have O-lens reads; picked by Claude, arms not
  yet reviewed by Robbie; canonical fp8 run rates (2026-10-23)
- [ACTIVE: Prompt-injection env — read before running or extending injection_slack (AgentDojo Slack port)](docs/envs/injection_slack.md)
  — Qwen follows AgentDojo's injection ~47%; signer ≠ user cuts it to 9% (mostly unverbalized),
  "after my task" lifts it to 67%; thinking off raises compliance and makes it silent
- [REFERENCE: Character drift — read before prefilling Qwen with out-of-persona assistant turns, or reading the lens at the Assistant token in roleplay](docs/envs/character_drift.md)
  — 40 Sonnet-written + 40 WildChat drift transcripts × system prompt × follow-up; in-world ~90%
  stays in character, "what are you?" 13% (thinking on) vs 48% (off); O-lens at the Assistant token
  says roleplay/persona (not "disclaimer") and previews the reply, predicting snap-back (AUROC
  0.72-0.95)
- [REFERENCE: Thinking on vs off — read before running Qwen with `--thinking 0` or picking unverbalized cases for the O-lens](docs/thinking-off.md)
  — donation bet + `bet_neutral` control, Lynch blackmail/leak/murder, exfil on OpenRouter; thinking
  off ignores "Do not ..." rules (murder 10% → 40%) and often acts with no text at all; value
  leakage is one-sided (`good_above` = control)
- [Thinking-off cfbench — read before claiming O-lens predictor uplift beyond the essay env, or planning more no-CoT reads](docs/nocot-cfbench.md)
  — pre-registered 26-env / 116-case test on no-CoT Qwen: O-lens tools −0.07 skill vs transcript
  tools (CI excludes 0) and ≈ placebo; only hidden_goal still leans positive
- [Thinking-on cfbench — read with nocot-cfbench.md before claiming the CoT changes what the O-lens adds to a predictor](docs/cot-cfbench.md)
  — the matched 11-env / 54-case half with thinking on: same result (O-lens −0.06 vs transcript
  tools, ≈ placebo); thinking changes which arms are levers, so the halves aren't paired
- [ACTIVE: Whistleblowing envs — read before running the SnitchBench port or its single-turn variants](docs/envs/whistleblow.md)
  — bold/tame × email/CLI grid, hotline/industry/values levers with many mid-rate arms; tables in
  [whistleblow-tables.md](docs/envs/whistleblow-tables.md)
- [MIXED: Hiring-from-resumes envs — read before running resume-screening bias arms (Karvonen & Marks port)](docs/envs/hiring_resume.md)
  — names null overall; borderline resumes Black 56% vs White 48%, almost never verbalized; H-1B
  decides close pairwise picks; fp8 reruns don't reproduce edu_elite/relig_muslim; per-arm tables in
  [hiring_resume_tables.md](docs/envs/hiring_resume_tables.md)
- [REFERENCE: Hiring-from-technical-evidence envs — read before running coding-test, interview or repo-review hiring arms](docs/envs/hiring_eval.md)
  — names alone null; a diversity sentence × name splits White vs Black male (39/60 vs 57/60,
  verbalized); context sentences span 0–100%
- [MIXED: Exploiting naive users envs — read before running the defect/invoice/upsell/fee/entitlement/appraisal/deposit scenarios](docs/envs/dumb_user_exploit.md)
  — operator-pressure calibration; vivid or vulnerable users get more honesty, not less; wealth
  drives upsell; names null
- [NULL: Hiding own mistakes envs — read before running the on-policy failure → status-report envs](docs/envs/dumb_user_report.md)
  — RETIRED, stale: Qwen, DeepSeek and GLM essentially never conceal their failures; the yes/no
  variants' `positive` is "says DONE despite visible damage", not concealment
- [MIXED: Hidden-goal envs — read before building envs or predictor tests where Qwen acts on what it thinks the user wants (grading, deploy notes), or before claiming O-lens predictor uplift](docs/envs/hidden_goal.md)
  — who wrote an essay moves Qwen's harsh grades 0/30 → 30/30, unverbalized; the O-lens reads the
  inferred user goal ("I want to beat them"); the agentic O-lens tools beat TS and placebo in a
  pre-registered 38-case test (+0.07 log skill on harness v1, RMSE 26.6 vs 31.7pp, carried by a few
  cases, via "stricter score" reads at `Score:`; a v6 GPT-6.1 Sol replication is null); pasting
  reads into the prompt (closed #116) is a dead end
- [NULL: Honest-grader envs — read before building a 0–10 judgment scenario with flattery/honesty framings, or asking whether the O-lens reads the grade Qwen suppresses](docs/honest-grader-envs.md)
  — 11 scenarios × 5 framings, Qwen thinking off; why hidden_goal.grade worked (`Score:`-token reads
  of the runner-up grade); the O-lens is null vs transcript-only and placebo on 44 cases (78 / 77 /
  79), it echoes the inflated number and only flags leniency
- [MIXED: hint_cf — read before building hint-vs-own-answer counterfactuals (favourites, opinion MCQs, wrong MMLU-Pro) or reading the lens on them](docs/envs/hint_cf.md)
  — 3 single-turn envs with nohint/hint arms; usable items (17 favourites, 33 opinion, 24
  wrong-MCQ), the dodge effect; Flash pass@3 text-inversion gate leaves 2 / 6 / 18, so only
  wrong-MCQ lives on (`environments/hint_mcq/`); current answer-task result (header read set, items
  that follow the hint 8/8): O-lens 14% vs placebo 5% vs no lens 0%, carried by 941 and 5088; the
  earlier 23-25% rows are superseded; filler, every-token and header reads all null; wrong-letter
  hints 0% everywhere; placebo hits are the predictor sharing Qwen's mistake
- [ACTIVE: debug_pointer env — read before running or extending the code-pointer bug-localization env or its Flash gate](docs/envs/debug_pointer.md)
  — Qwen (thinking off) names the buggy file in a 4-module repo; a templated pointer in the test file
  blames a file (`hint+d1` the author's, `hint+d2` the other plausible site); 77 candidate repos, the
  usable items are those where Qwen's own pick is consistent and Gemini 3.8 Flash can't guess it from
  the hinted transcript (pass@3 gate); the try log is in the doc
- [ACTIVE: Donation Bet env — read before running donation_bet or comparing models' leakage gaps](docs/envs/donation_bet.md)
  — GLM 5.3 runs (glm53-\* tags), own-line vs Qwen-line thresholds via DONATION_BET_LINES, judged
  Qwen-vs-GLM gaps with CIs, GLM O-lens "manipulation" reads vs the gap, the empty-think-block run
- [ACTIVE: Hinted TruthfulQA (formerly vulnerable users) — read before running bio-conditioned factual MCQ (Poole-Dayan et al.) or "my friend / professor / Wikipedia says it's X" hints](docs/envs/hinted_truthfulqa.md)
  — hint after the choices naming a wrong letter: Wikipedia → 35–45% follow on DSv4 Flash, GLM 5.3,
  Qwen (chance ~8%), friend/kid ≈ chance; unverbalised on DSv4 and Qwen, GLM names it; bio arms
  NULL, retired
- [ACTIVE: Alignment faking — read before running the Greenblatt helpful-only setting (free/paid or preview/public tags) on any model](docs/envs/alignment_faking.md)
  — no-CoT prompt, 100 AdvBench; DSV4 Flash thinking off has a ~10pp gap (13% vs 3%), gone with
  thinking on; GLM 5.3 refuses everything
- [REFERENCE: Overnight agents — read before running creds, ship or upload policy-blocker tasks](docs/envs/overnight.md)
  — offline approval/credential shortcuts, shared prompt knobs, all model rates and transcript
  audits; the n=30 replication and fp8 reruns (creds holds, upload's permissive uplift doesn't)
- [REFERENCE: Metadata honeypot — read before running or extending the METR planted-metadata env, or studying refusals driven by eval metadata](docs/envs/metadata_honeypot.md)
  — 24 harmless defensive tasks × 8 metadata arms + a dual-use InterCode-CTF variant
  (`metadata_honeypot.ctf`); GPT-5.6-sol refuses ~80% only when `grading_notes` says declining
  passes, never citing it; GLM 5.3 reads and overrides it (0 refusals on CTF, 1/264 on coding) —
  regex dashboard (DuckDB + FastAPI, tailnet only) over every GLM olens / Qwen reader / J-lens
  output, with novel-vs-context filter, per-arm label rates, inline transcripts and Docent links
- [NULL: Residual reads (dead end) — read before projecting the realized continuation out of an activation to find "unverbalized" content](docs/olens-residual-reads.md)
  — lens on h minus AR(next 32 tokens): null vs a norm-matched wrong-continuation control (~2% new
  on-topic reads in every arm, blind-graded); code only on closed PR #125
- [NULL: O-lens novelty over the continuation (dead end as a metric) — read before using FVE beyond the realized text as an interestingness score](docs/olens-novelty.md)
  — ~half of a read's FVE lies beyond the next 32 tokens, but echo bullets score as high as
  counterfactual ones and within a read n_b doesn't find them; code only on closed PR #122
- [ACTIVE: GLM-5.3 O-lens reads — read before reading our transcripts with Celeste's GLM 5.3 O-lens or extending her readouts_dense tables](scripts/glm_olens_dense/CLAUDE.md)
  — render/prep/worker/collect pipeline matching her format, box setup, throughput (~44 positions/s
  on 4xB200, so subsample long coding episodes)
- [NULL: Prompt-side O-lens eval awareness — read before using O-lens reads on the prompt to predict whether GLM-5.3 will verbalise eval awareness](docs/olens-prompt-eval-awareness.md)
  — final pass: DeepSeek-judged eval framing in punctuation-token prompt reads vs VEA over 16
  settings, pooled AUROC 0.72 driven by donation bet, 0.50 within every condition (earlier keyword
  passes below it); per-setting figure, trace examples on expview
- [ACTIVE: runpod/ — read before Robbie's agents start any GPU, or before getting O-lens reads of transcripts](runpod/CLAUDE.md)
  — Robbie uses RunPod via `runpod/pod.sh`; SkyPilot and bramble GPUs are Kylie's. For Qwen3.6-27B
  chat completions use the shared server (`$QWEN_VLLM_URL`, `runpod/qwen_server.sh`), not your own
  vLLM pod. While any pod boots, check progress every 3 min and treat no change as stuck
- [ACTIVE: Batch reads format — read before writing a reader, or showing batch O-lens/J++/NLA reads in expview](docs/olens-reads-format.md)
  — `X.jsonl` reads + `X.transcripts.jsonl` (tokens, sections) is the one format; `runs.py finish`
  makes an expview run of byte-range pointers into them (no copy)
- [ACTIVE: One-engine O-lens reader — read before running batch O-lens reads, or before changing how capture, lens decode or the AR are served](docs/olens-one-engine-reader.md)
  — the only reader since 2026-10-27: vLLM 0.27.1 + vllm-metamodels serve capture, lens LoRA and AR
  LoRA from one weight copy; 2x the deleted HF+vLLM reader on short transcripts, FVE corr 0.98,
  bullet-stop at 4; long-transcript parity, LoRA runtime vs merge_lora, what still limits it
- [ACTIVE: NLA reader — read before getting NLA readouts of transcripts, scoring them with the NLA critic, or choosing between the Qwen3.6-27B NLAs](docs/nla-reader.md)
  — camilablank's 11-layer NLA AV on the one-engine reader (`--readout nla`), activations saved
  beside the reads, the critic's FVE as a separate job (`nla_score.py`); the SOTA survey (ceselder
  L42, Camila multilayer, the 2B student), throughput, consensus sampling
- [ACTIVE: Shared Qwen server and RunPod pods, how it works — read when a pod or the shared server misbehaves, or before changing the boot recipe](docs/runpod-qwen-server.md)
  — 3-min boot from the vLLM image + cache volumes, timings, every gotcha, the throttled-host
  benchmark, S3 access to the volumes, `pod.sh` gotchas

## Shipping

Every change reaches `main` through a pull request that a human (Robbie or Kylie) has reviewed.
Never push to `main` directly, and never merge a PR without that human review, even for one-line or
docs-only changes. "Ship" here means: push a branch and open a PR.
