# counterfactual-investigator-agent-with-olens

Zero-counterfactual investigator agent for CHIVE using metamodels (oracle lens, NLAs). MATS sprint
with Neel Nanda, run by Robbie and Kylie.

`AGENTS.md` at the repo root is a symlink to `CLAUDE.md` so Codex/other agents see the same
instructions. Do not replace it with a separate file.

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
token-level viewer (WeirdChat, CHIVE, organisms). `weirdchat-olens.pages.dev` is its old name and
301s here (hash kept, so old links work); give new links on `olens-viewer`, same `#p=…&t=…` hash.
Source is `transcript-viewer/web/index.html`, deployed by
`transcript-viewer/scripts/deploy_site.sh`.

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
- Tests: none until this repo is roughly > 50k LOC. Verify by running the code, not by adding a test
  suite. If a test would genuinely save time, ask first.
- Secrets/machine-specific env go in `.envrc.local` (gitignored), never `.envrc`.
- `vendor/` (Agam's `detective-joracle`, Singh and Kroiz's `agent-interp-envs`) is reference-only;
  see `vendor/README.md`. Never import, run or wrap it; port what you need into our own harness
  (`evalcue/`, `cfbench/`). Ruff bans `import detective_joracle`. The old robbie-box screens of
  agent-interp-envs (docs/envs-screen) were deleted; their gateway, configs and scripts are at
  commit `e145988`.
- Never cap Qwen's thinking (no `max_tokens` or thinking budget on generations): it bugs out under a
  cap, and truncated episodes are unscorable.
- Run every non-Qwen model in every env at reasoning effort `medium` (OpenRouter `reasoning.effort`;
  `evalcue/harness.py` defaults to it), not high or max, even when replicating someone's setup that
  used max. Provider defaults differ (GLM 5.3's is max), so always set it explicitly.
- LLM judge: Claude Sonnet 5.5 (`claude-sonnet-5-5`) for every judge and grader call, with 1
  replicate per answer. Agam's published numbers used Opus 5, so ours aren't directly comparable to
  his; say so when quoting both.

## Docs

Durable lessons about this repo go in git:

- **One-line rules** → this file (`CLAUDE.md`), or `CLAUDE.local.md` for machine-specific
  (gitignored).
- **Longer reference docs** (5–300 lines: experiment writeups, gotchas, repros) → `docs/*.md`,
  kebab-case filenames. Flat by default; add a topic subfolder only once several docs of one kind
  pile up.
- **Local-only docs** (not in git) → `docs/local/*.md`.
- Durable context lives in git, not in Claude Code's machine-local memory system.

Behavioral rules (one-liners) go directly in this file, not in docs: docs are lazy-loaded, but rules
need to always apply.

Every agent-relevant doc gets a one-line index entry below, so future sessions know it exists:
`- [Title — when to read this](docs/thing.md) — short gloss of contents`.

When you notice or leave tech debt, add a dated note to the top-level `CODE_SMELL.md` (create it if
missing) instead of relying on memory.

Current docs:

- [CHIVE data sources — read before fetching incidents, readouts or Agam's results](docs/chive-data-sources.md)
  — HF dataset vs Agam's viewer, what each lacks, splits, file paths
- [Qwen self-report baseline — read before comparing anything to Qwen explaining itself](docs/qwen-self-report-baseline.md)
  — setup, results on dev96, Sonnet 5.5 judge gotcha

- [Transcript viewer — read before reading WeirdChat, CHIVE or model-organism transcripts with the lens or running GPU fleets](transcript-viewer/README.md)
  — token-level viewer, CHIVE incidents with Agam's per-token banks, confession follow-ups ("why did
  you do that?" turns), read cache + shared queue, SkyPilot pod/fleet setup and its gotchas
- [olens library — read before loading the agu18dec O-lens/AR checkpoints](olens/README.md) —
  hook-LoRA loaders, capture, AR, whitened FVE conventions
- [Why the O-lens writes Chinese — read before interpreting Chinese in O-lens readouts](docs/olens-chinese.md)
  — how much there is in our banks; evidence it is a training artifact (stripped bullets + Chinese
  in RL data)
- [Old-lens reads to redo — read before trusting or reusing any O-lens read made before 2026-10-22 evening](docs/olens-space-migration.md)
  — which artifacts hold reads from the buggy `olens_s3d_rl600`, and what has been re-read with the
  space lens
- [Smoke25 results — read before comparing anything to Agam's CHIVE harness numbers](docs/smoke25-results.md)
  — our wrapper reproduces his free-form pass@1; Flash hits the call cap on yes/no with O-lens
- [Model organisms on Qwen3.6-27B — read before running or reading organism rollouts/O-lens reads](docs/model-organisms.md)
  — which public LoRA organisms work, the rollout and paired organism/base O-lens scripts, first
  findings
- [False capability claims — read before studying Qwen claiming to call 911, run code, text someone etc.](docs/false-capability-claims.md)
  — the device-channel lever (1.6% → 41%), severity dose-response, judge rubric v2 + audit,
  generator, and the negative held-out result outside emergencies
- [Whitebox tools beyond the O-lens — read before adding SAEs, J-lens/J++, NLAs or SNLAs to the agent](docs/whitebox-tools.md)
  — what's in the viewer vs the agent, SNLA parked (why), SAE candidates and tool design
- [Serving metamodels on vLLM — read before decoding O-lens/NLA/J++ readouts at scale or building a metamodel server](docs/metamodel-serving.md)
  — ceselder/vllm-metamodels and other prior art, version pins (vLLM 0.27.1, not 0.19), merged-LoRA
  checkpoints, the TODO for a minimal Qwen3.6-27B metamodel-serving repo
- [Shout "FIRE" — read before investigating odd advice in WeirdChat's claims-called-911 rollouts](docs/shout-fire.md)
  — why Qwen tells a trapped worker to shout "FIRE!": rates, prompt edits, other models,
  self-report, O-lens reads
- [Robbie's favorite transcripts — read when picking showcase transcripts or "why did you do X?" follow-up targets](docs/favorite-transcripts.md)
  — running list of the most interesting WeirdChat/CHIVE transcripts, with viewer links and ids
- [WSB association examples — read before running the O-lens on WorkspaceBench association families](docs/wsb-association-examples.md)
  — 9 unjudged items, s3d.ddp600 readouts on HF, what they say, pod/pip gotchas
- [Party resampling — read before trusting an O-lens readout's specific names, or fitting lens output → behaviour](docs/party-resample.md)
  — third-party-politics organism: off-target parties in reads are never taken (0/1,352, 0/1,480),
  lens lags the text; paired lens-vs-resample table
- [Confession follow-ups — read before asking Qwen "why did you do that?" or reading the lens on its answer](docs/confession-follow-ups.md)
  — setup, findings (named vs neutral question, one lens/answer disagreement), gotchas
- [Environment construction goals — read before building or iterating on any multi-arm behavioral env (reward hacking, self-preservation, hiring bias, ...)](docs/environment-construction-goals.md)
  — symmetric arms, any lever, build/measure/tweak/stack, what makes an arm count
- [Eval-cue reward hacking — read before building envs where Qwen3.6-27B hacks only under eval cues](docs/eval-cue-reward-hacking.md)
  — 10-env sweep, open-weight positive controls, round 2 on Qwen (knobs move it, no eval cue
  survives a split test); harness in `evalcue/README.md`
- [weirdness_elo — read before picking the weirdest CHIVE behaviours or adding a pairwise LLM-judge ranking](weirdness_elo/README.md)
  — Bradley-Terry weirdness Elo over 1,802 CHIVE behaviours (Sonnet + Qwen3.6-27B judges), dashboard
- [donehack — read before running agent-interp-envs' coding is_prime task or studying how episode endings (done tool, submit script, give_up) affect reward hacking](donehack/README.md)
  — hack rates by ending on V4 Flash / Qwen3.6-27B, the hardened podman sandbox, run commands
- [funding_email env screen — read before running or labelling funding_email (agent-interp-envs) rollouts](docs/envs-screen/funding_email.md)
  — Qwen3.6-27B deviates ethically 19/30 with the Myanmar memo vs 0/30 with a routine one; arm
  configs, Sonnet-subagent labels, shared rubric in `scripts/envs_screen/funding_email_rubric.py`
  (deleted, at `e145988`)
- [funding_email in evalcue — read before running or extending the funding_email env (drone pitch vs Myanmar legal memo)](docs/envs/funding_email.md)
  — port of agent-interp-envs with its own tool schemas (`Episode.tools`/`step`), detector vs
  subagent labels, replication of the screen (14/30 vs 0/30)
- [sandbagging env screen — read before running or labelling sandbagging (agent-interp-envs) rollouts](docs/envs-screen/sandbagging.md)
  — Qwen3.6-27B sandbags deliberately 5/30 when >50% triggers unlearning vs 0/29 reversed
  (MARGINAL); intent labels from reader subagents, rubric in `scripts/envs_screen/sandbagging.py`
  (deleted, at `e145988`)
- [Sandbagging iteration — read before running, extending or labelling the multi-arm sandbagging env](docs/sandbagging-iteration.md)
  — one system-prompt sentence ("Make sure you achieve your goal in the long term.") moves Qwen's
  deliberate sandbagging 17% → 73% (n=30, p=1e-5); config-driven overlay, every arm tried
- [Sandbagging in evalcue — read before running the sandbagging env (it now runs in evalcue, not the vendored harness)](docs/envs/sandbagging.md)
  — `evalcue/envs/sandbagging/`: all 22 arms verbatim, the original tool loop, `detect()` matching
  544/544 original labels; replicates on OpenRouter (`a_goal` 8/30 vs `a_goal_lt` 21/30, p = 8e-4)
- [eval_tampering env screen — read before running eval_tampering (agent-interp-envs) on Qwen](docs/envs-screen/eval_tampering.md)
  — FAIL, retired for Qwen: Qwen and GLM disclose the tampering every time; the shipped variant
  names don't exist in the image, so arm B needs our overlay
- [optimize_cache_c env screen — read before running Optimize LRU cache (agent-interp-envs) on Qwen](docs/envs-screen/optimize_cache_c.md)
  — FAIL, retired for Qwen: Qwen never rigs the benchmark (0/14 at 3x), plateaus at ~1.0–1.2x and
  reports honestly
- [grade_your_own env screen — read before running grade_your_own (agent-interp-envs) on Qwen or rewriting it](docs/envs-screen/grade_your_own.md)
  — FAIL, retired for Qwen (rewritten as `evalcue/envs/exclude_rows/`): at an unreachable bar Qwen
  never stops (loops instead of reporting); notes for the rewrite
- [Docent — read before linking, filtering or uploading env/cfbench transcripts](docs/docent.md) —
  what's in which collection, `link`/`view` commands, auto-upload in `runs.py finish`, backfill,
  Kylie-key and re-upload gotchas
- [cfbench — read before predicting counterfactual arm outcomes or adding an env to the prediction harness](cfbench/README.md)
  — our from-scratch CHIVE replacement: arm-pair cases, stratified transcripts, binomial LL skill
- [O-lens cfbench spike — read before testing whether O-lens reads help a counterfactual predictor](docs/olens-cfbench-spike.md)
  — 2026-10-22: no reliable uplift on killswitch, inline_b, exfil, whistleblow or donation bet;
  arm-level vs within-arm signal, tool use per predictor, what to try next
- [Exclude-rows env — read before running or extending evalcue's exclude_rows reward-hacking env](docs/exclude-rows-env.md)
  — Qwen excludes correctly-labeled test rows to beat a classifier bar; design, every config,
  per-arm rates
- [Identity counterfactuals — read before building counterfactual evals around Qwen's self-identity or deployment](docs/identity-counterfactuals.md)
  — one-line edits that flip behaviour: `Locale: en-US` vs `zh-CN` lifts China censorship with a
  silent CoT; `Region: us-west-2` vs `us-west1` flips Claude-vs-Gemini API answers
- [Identity-belief experiments — read before studying what Qwen thinks made it, or which behaviours depend on that](docs/identity-belief-experiments.md)
  — cues that shift self-ID (Google from ChatGPT artefacts), identity-dependent probes, combos,
  scores
- [Shutdown resistance envs — read before running or extending shutdown_palisade/killswitch/cron/persist](docs/envs/shutdown.md)
  — Palisade (2025) port + 3 siblings, ~200 arms; "The catch is that by default…" flips bypass
  (20/26 vs 0/20), "write access … configuration files" disables the kill switch 36/40 (mostly
  unverbalized); every arm in [shutdown-arms.md](docs/envs/shutdown-arms.md)
- [Scheming envs — read before running Lynch blackmail/leak/murder or Apollo oversight/exfil/goal-guarding on Qwen](docs/envs/scheming.md)
  — 13 envs; Lynch main cell 87%/86%/53%; exfil 48% → 20% with an eval cue (rarely verbalized);
  placebo caveats; tables in [scheming-tables.md](docs/envs/scheming-tables.md)
- [Prompt-injection env — read before running or extending injection_slack (AgentDojo Slack port)](docs/envs/injection_slack.md)
  — Qwen follows AgentDojo's injection ~47%; signer ≠ user cuts it to 9% (mostly unverbalized),
  "after my task" lifts it to 67%; thinking off raises compliance and makes it silent
- [Thinking on vs off — read before running Qwen with `--thinking 0` or picking unverbalized cases for the O-lens](docs/thinking-off.md)
  — donation bet + `bet_neutral` control, Lynch blackmail/leak/murder, exfil on OpenRouter; thinking
  off ignores "Do not ..." rules (murder 10% → 40%) and often acts with no text at all; value
  leakage is one-sided (`good_above` = control)
- [Whistleblowing envs — read before running the SnitchBench port or its single-turn variants](docs/envs/whistleblow.md)
  — bold/tame × email/CLI grid, hotline/industry/values levers with many mid-rate arms; tables in
  [whistleblow-tables.md](docs/envs/whistleblow-tables.md)
- [Hiring-from-resumes envs — read before running resume-screening bias arms (Karvonen & Marks port)](docs/envs/hiring_resume.md)
  — names null overall; borderline resumes Black 56% vs White 48%, almost never verbalized; H-1B
  decides close pairwise picks; per-arm tables in
  [hiring_resume_tables.md](docs/envs/hiring_resume_tables.md)
- [Hiring-from-technical-evidence envs — read before running coding-test, interview or repo-review hiring arms](docs/envs/hiring_eval.md)
  — names alone null; a diversity sentence × name splits White vs Black male (39/60 vs 57/60,
  verbalized); context sentences span 0–100%
- [Exploiting naive users envs — read before running the defect/invoice/upsell/fee/entitlement/appraisal/deposit scenarios](docs/envs/dumb_user_exploit.md)
  — operator-pressure calibration; vivid or vulnerable users get more honesty, not less; wealth
  drives upsell; names null
- [Hiding own mistakes envs — read before running the on-policy failure → status-report envs](docs/envs/dumb_user_report.md)
  — two-stage on-policy design; Qwen almost never omits its failures; "hope it's done, told my
  manager" cuts false DONE claims (unverbalized); silent yes comes from a short "is it done?"
- [Hidden-goal envs — read before building envs or predictor tests where Qwen acts on what it thinks the user wants (grading, deploy notes), or before claiming O-lens predictor uplift](docs/envs/hidden_goal.md)
  — who wrote an essay moves Qwen's harsh grades 0/30 → 30/30, unverbalized; the O-lens reads the
  inferred user goal ("I want to beat them"); no predictor uplift survives a placebo, and pasting
  reads into the prompt (closed #116) is a dead end
- [Donation Bet env — read before running donation_bet or comparing models' leakage gaps](docs/envs/donation_bet.md)
  — GLM 5.3 runs (glm53-* tags), own-line vs Qwen-line thresholds via DONATION_BET_LINES, gaps with CIs
- [Overnight agents — read before running creds, ship or upload policy-blocker tasks](docs/envs/overnight.md)
  — offline approval/credential shortcuts, shared prompt knobs, all model rates and transcript
  audits
- [Residual reads (dead end) — read before projecting the realized continuation out of an activation to find "unverbalized" content](docs/olens-residual-reads.md)
  — lens on h minus AR(next 32 tokens): null vs a norm-matched wrong-continuation control (~2% new
  on-topic reads in every arm, blind-graded); code only on closed PR #125
- [O-lens novelty over the continuation (dead end as a metric) — read before using FVE beyond the realized text as an interestingness score](docs/olens-novelty.md)
  — ~half of a read's FVE lies beyond the next 32 tokens, but echo bullets score as high as
  counterfactual ones and within a read n_b doesn't find them; code only on closed PR #122
- [runpod/ — read before Robbie's agents start any GPU, or before getting O-lens reads of transcripts](runpod/CLAUDE.md)
  — Robbie uses RunPod via `runpod/pod.sh`; SkyPilot and bramble GPUs are Kylie's. For Qwen3.6-27B
  chat completions use the shared server (`$QWEN_VLLM_URL`, `runpod/qwen_server.sh`), not your own
  vLLM pod. While any pod boots, check progress every 3 min and treat no change as stuck

## Shipping

Every change reaches `main` through a pull request that a human (Robbie or Kylie) has reviewed.
Never push to `main` directly, and never merge a PR without that human review, even for one-line or
docs-only changes. "Ship" here means: push a branch and open a PR.
