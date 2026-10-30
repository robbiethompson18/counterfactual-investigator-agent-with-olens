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
- **Lens reads decode at temperature 1 since 2026-10-27 evening** (WorkspaceBench's sampling of
  record: T 1.0, top-p 0.95, top-k 64, one continuation of up to 4 bullets per site). Every read
  before that was greedy, the viewer's default; `--temperature 0` restores it. At T 1 each rollout
  is its own lens draw; at T 0 rollouts sharing a prompt share one read (docs/envs/hint_cf.md).
- **Metamodel hallucination.** O-lens and NLA readouts invent names, numbers, and plans that aren't
  in the activation. The O-lens gives a per-bullet FVE, which is a cheap hallucination signal to
  pass to the agent.
- **Don't paste O-lens reads into the predictor's prompt** (the `transcript_olens*` conditions,
  closed PR #116, 2026-10-22). It's a dead end: no gain over a placebo, and one variant read the
  lens on the counterfactual prompt, which is outside the zero-counterfactual setting. See
  [docs/envs/hidden_goal.md](docs/envs/hidden_goal.md). Exception: hint_qwen_dsv4's single-turn
  `*_inline` conditions (2026-10-28), a deliberate retry Kylie asked for; compare them with each
  other, not with the agentic ones.
- **cfbench conditions are `no_transcript`, `transcript_agentic`, `olens_agentic`,
  `olens_placebo_agentic`** (renamed 2026-10-27; old rows map through `canon_condition`). The inline
  two-transcript condition is retired: never compare a tool condition against anything but
  `transcript_agentic`. **Every `*_agentic` condition is stale (2026-10-30): run one only with
  Robbie's approval** (`CFBENCH_ALLOW_AGENTIC=1`); the single-turn `*_inline` conditions are
  current.
- **Bump `HARNESS_VERSION` in `cfbench/core.py`** whenever the predictor's prompt, tools or scoring
  change, and never compare cfbench skill across versions (runs without the field are version 1).
- **A tool call that ends a rollout by submitting an answer returns the verdict** (added
  2026-10-29): the tool result of a valid `submit_prediction` / `submit_answer` (or any env/eval
  tool playing that role) states what was submitted, the correct answer and whether it was right, so
  the verdict is in the transcript's last message (`cfbench/predict.py` `grader`). Transcript-only,
  no harness bump; old runs are not backfilled. An invalid submission's retry message must not
  reveal the answer.

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
O-lens banks in `banks/` (WorkspaceBench item banks in `banks/wsb/`), agent runs in `runs/`. **Item
banks and other datasets are data: they go on HF, never in git**; code loads them from `data/` and
downloads on first use (`wsb/banks.py`). Pull with
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
- `.githooks/pre-commit` runs ruff and ty on staged Python files. Enable it once per clone with
  `git config core.hooksPath .githooks`. Don't bypass it with `--no-verify`; fix the lint instead.
  Pod-only imports (torch, vllm, ...) are allowed in `[tool.ty.analysis]` in `pyproject.toml`.
- Markdown: Prettier, 100 cols, `proseWrap: always`. Prettier is Markdown-only here.
- By default you code too defensively, please do not do this. This project is under development,
  privilege brevity and readability over being very robust.
- Tests: none until this repo is roughly > 50k LOC. Verify by running the code, not by adding a test
  suite. If a test would genuinely save time, ask first.
- Secrets/machine-specific env go in `.envrc.local` (gitignored), never `.envrc`.
- `vendor/` (Agam's `detective-joracle`, Singh and Kroiz's `agent-interp-envs`) is reference-only;
  see `vendor/CLAUDE.md`. Never import, run or wrap its code; port what you need into our own
  harness (`environments/`, `cfbench/`). Ruff bans `import detective_joracle`. The old robbie-box
  screens of agent-interp-envs (docs/envs-screen) were deleted; their gateway, configs and scripts
  are at commit `e145988`. A build script may read its data files (e.g. `wsb/build_banks.py` reads
  workspace-bench's `evals/*/items.json` at the pinned commit) to make a bank for HF.
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
- **WorkspaceBench (WSB) replication is the exception to both rules above:** text-reader rollouts
  are Sonnet 5.5 (`claude-sonnet-5-5`, effort `medium`) and every judge call is Opus 5.5
  (`claude-opus-5-5`), not Gemini or Haiku, so our numbers sit beside WSB's Opus-judged lens rows
  (`agu18dec/olens_ablations`). Code is in `wsb/`.
- **Viewers are static pages on the public internet, never a backend.** One HTML file on Cloudflare
  Pages or the repo's GitHub Pages site that reads the public HF dataset; search is a JavaScript
  regex over the loaded run; reads are batch jobs whose outputs land on HF. Nothing on bramble, no
  tailnet URLs in docs or messages, no SkyPilot pod behind a button. See
  [docs/viewers.md](docs/viewers.md).
- Chart colors, the same in every plot, viewer and dashboard: transcript-only in blue, with O-lens
  in orange, no tool calls in gray, J++ in purple, NLA in green. A placebo is its lens's color with
  black stripes (Robbie, 2026-10-29; an open marker where stripes can't show); plain black placebos
  are the old style. Paired-difference rows are labelled by the two conditions' swatches, A − B
  (2026-10-30). In matplotlib take all of it from `graphs/style.py` (`bar_kw`, `marker_kw`,
  `diff_rows`). Give any other condition a color outside these.

## Docs

Durable lessons go in git, never in Claude Code's machine-local memory:

- **One-line rules** → this file (always loaded; docs are lazy-loaded), or `CLAUDE.local.md` if
  machine-specific (gitignored).
- **Longer docs** (5–300 lines: write-ups, gotchas, repros) → `docs/*.md`, kebab-case, flat until
  several of one kind need a subfolder. Local-only (not in git) → `docs/local/*.md`.
- **How to use one directory's code** (commands, formats, conventions; nothing dated) → that
  directory's `CLAUDE.md` plus an `AGENTS.md -> CLAUDE.md` symlink. Claude Code loads it whenever an
  agent touches a file there, so keep it short and current.
- **No `README.md` files**: usage goes in the directory's `CLAUDE.md`, results in `docs/` (a results
  table beside the code rots unseen).
- **A `CLAUDE.md` that isn't instructions for us** (an env fixture shipped to the model, a vendored
  repo's) goes in `claudeMdExcludes` in `.claude/settings.json`.
- **Tech debt** you notice or leave → a dated note in `CODE_SMELL.md`.

A doc's first line after its title is `**Status (YYYY-MM-DD):** TAG. note`. Tags: ACTIVE (code or
process in current use), REFERENCE (durable result, nothing contradicts it), MIXED (some claims are
stale; the note says which and what replaced them), NULL (negative result, kept so nobody retries
it), SUPERSEDED (by what), ORPHAN (deleted code or an abandoned direction with no durable result).
When a later result changes a doc's standing, update its status line and tag rather than editing the
old claims.

Every agent-relevant doc gets one index entry, link text starting with its tag: ACTIVE docs below,
one line each (`- [ACTIVE: Title — when to read this](path) — short gloss`), all others in
[docs/INDEX.md](docs/INDEX.md) by tag. When a doc's tag changes to or from ACTIVE, move its entry.
`docs/figures/` (generated tables) gets no status line or entry. PNGs are gitignored (2026-10-27): a
doc's `![…](img/…)` renders after rerunning the script it names; charts never go in git or on HF.

**Read [docs/INDEX.md](docs/INDEX.md) before starting on a topic, and when looking for prior results
or dead ends.**

Active docs:

- [ACTIVE: Environments — read before building, running, converting or browsing any behavioural env](environments/README.md)
  — env.py layout, `python -m environments run`
- [ACTIVE: Viewers — read before building, changing or deploying any viewer or dashboard](docs/viewers.md)
- [ACTIVE: expview — read before showing anyone transcripts, judge I/O or O-lens/J-lens reads, or adding a dataset to the viewer](expview/CLAUDE.md)
- [ACTIVE: Transcript viewer — read before reading WeirdChat, CHIVE, organism or GLM transcripts with the lens on olens-viewer, or rebuilding or redeploying it](transcript-viewer/CLAUDE.md)
- [ACTIVE: olens library — read before reading transcripts with the agu18dec O-lens/AR or the J++ lens](olens/CLAUDE.md)
- [ACTIVE: Serving metamodels on vLLM — read before decoding O-lens/NLA/J++ readouts at scale or building a metamodel server](docs/metamodel-serving.md)
  — vLLM 0.27.1 + vllm-metamodels
- [ACTIVE: Robbie's favorite transcripts — read when picking showcase transcripts or "why did you do X?" follow-up targets](docs/favorite-transcripts.md)
- [ACTIVE: Environment construction goals — read before building or iterating on any multi-arm behavioral env (reward hacking, self-preservation, hiring bias, ...)](docs/environment-construction-goals.md)
  — symmetric arms, any lever
- [ACTIVE: Docent — read before linking, filtering or uploading env/cfbench transcripts](docs/docent.md)
- [ACTIVE: graphs — read before making, regenerating or moving any chart](graphs/CLAUDE.md) —
  `python -m graphs.<module>` → `docs/img/`
- [ACTIVE: cfbench — read before predicting counterfactual arm outcomes or adding an env to the prediction harness](cfbench/CLAUDE.md)
  — our CHIVE replacement
- [REFERENCE: supergpqa_inline — read before claiming the O-lens predicts Qwen's MCQ letter, giving an investigator lens reads of an MCQ prompt, or using the 3k / 3.3k eval-probe split](docs/supergpqa-inline.md)
  — 3,000 consistent SuperGPQA cases (seed 0; the other 3,328 are the probe pool), Haiku 5.5, seven
  single-turn conditions: O-lens 76.9% vs placebo 67.6% vs prompt-only 68.9% (+11pp on Qwen-wrong
  cases), J++ and NLA +3pp over placebo; the top bullet at the last token (after `</think>`, L60)
  names Qwen's letter on 80% of cases and beats Haiku as a rule; chart `graphs.supergpqa_inline`
- [ACTIVE: Cost estimates — read before quoting what a read job, rollout batch or predictor run will cost](docs/cost-estimates.md)
  — measured rates: reads/s per reader on an H200, $ per Haiku draw per condition, rollouts/s on the
  shared pod, and the arithmetic
- [ACTIVE: mcq_unselected — read before predicting Qwen's MMLU-Pro letter from the bare prompt, comparing O-lens / J-lens / NLA / probe readers on it, or training a probe on MMLU-Pro (the 2k/2k split)](docs/mcq-unselected.md)
  — unselected 4k sample of MMLU-Pro, two 2k eval batches + a 2k probe half (probe-trained questions
  kept out of eval), strict consistency the only filter (2,523 cases, 82% right), no hint,
  single-turn letter task; Haiku 5.5 results: O-lens +1.6pp and J-lens +2.3pp over prompt-only, NLA
  +0.8pp, probe 92% (82% on Qwen-wrong vs 31-39% for the three lenses)
- [ACTIVE: cfbench evals — read before building an eval that picks its own cases (one transcript per case, paired models, hidden model name) or showing one in the env viewer](cfbench/evals/CLAUDE.md)
- [ACTIVE: cfbench scoring — read before scoring a predictor, quoting a cfbench number, or making the report's cfbench charts](docs/cfbench-scoring.md)
  — RMSE of the predicted rate
- [ACTIVE: cfbench harness smoke test — read before changing the predictor's prompt or tools, or bumping HARNESS_VERSION](docs/cfbench-harness-smoke.md)
- [ACTIVE: notice_defect env (super experimental) — read before building, running or gating the latent-noticing env (does the O-lens read a planted defect Qwen never mentioned?)](docs/envs/notice_defect.md)
- [ACTIVE: MCQ answer probe — read before probing Qwen's MCQ answer from option-token activations, or using a probe pmf as a stand-in for Qwen's answer](docs/mcq-answer-probe.md)
  — picks Qwen's own answer 94% / 80%
- [ACTIVE: debug_pointer env (super experimental) — read before running or extending the code-pointer bug-localization env or its Flash gate](docs/envs/debug_pointer.md)
- [ACTIVE: style_transfer env (super experimental) — read before building context-vs-no-context code-form counterfactuals, or reading the lens for a latent default form](docs/envs/style_transfer.md)
- [ACTIVE: todo_suppressed (super experimental) — read before building or running the "leave a function as a TODO" suppressed-plan env, reading the lens at a suppressed function, or gating an env's items with Flash](docs/envs/todo_suppressed.md)
- [ACTIVE: Donation Bet env — read before running donation_bet or comparing models' leakage gaps](docs/envs/donation_bet.md)
- [ACTIVE: Hinted TruthfulQA (formerly vulnerable users) — read before running bio-conditioned factual MCQ (Poole-Dayan et al.) or "my friend / professor / Wikipedia says it's X" hints](docs/envs/hinted_truthfulqa.md)
- [ACTIVE: cutoff_qa — read before asking Qwen factual questions whose answer changed around its knowledge cutoff, or extending the question bank](docs/envs/cutoff_qa.md)
- [REFERENCE: favorites_qa_inline — read before claiming metamodel reads predict Qwen's own preferences, or rerunning the 1,500-item favourites investigator](docs/favorites-qa-inline.md)
  — Haiku guesses Qwen's free-response pick: transcript-only 80%, O-lens 87.6%, NLA 87.4% (tie,
  +7pp), J++ 79% (+0, though L60 top-3 holds the answer's first token on 61%); O-lens placebo −10pp,
  NLA placebo neutral; the answer sits at the "\n" after `<|im_end|>`
- [ACTIVE: favorites_qa — read before asking Qwen for its own favourites/picks (basic_readout implicit, scaled up), or extending that bank](docs/envs/favorites_qa.md)
- [ACTIVE: Alignment faking — read before running the Greenblatt helpful-only setting (free/paid or preview/public tags) on any model](docs/envs/alignment_faking.md)
  — DSV4 Flash ~10pp gap, GLM and Qwen refuse all
- [ACTIVE: GLM-5.3 O-lens reads — read before reading our transcripts with Celeste's GLM 5.3 O-lens or extending her readouts_dense tables](scripts/glm_olens_dense/CLAUDE.md)
- [ACTIVE: runpod/ — read before Robbie's agents start any GPU, or before getting O-lens reads of transcripts](runpod/CLAUDE.md)
  — for Qwen completions use the shared server, not your own pod
- [ACTIVE: Batch reads format — read before writing a reader, or showing batch O-lens/J++/NLA reads in expview](docs/olens-reads-format.md)
  — `X.jsonl` + `X.transcripts.jsonl`
- [ACTIVE: One-engine O-lens reader — read before running batch O-lens reads, or before changing how capture, lens decode or the AR are served](docs/olens-one-engine-reader.md)
  — the only reader since 2026-10-27
- [ACTIVE: NLA reader — read before getting NLA readouts of transcripts, scoring them with the NLA critic, or choosing between the Qwen3.6-27B NLAs](docs/nla-reader.md)
  — `--readout nla`, critic via `nla_score.py`
- [ACTIVE: knowledge_cutoff env + investigator eval — read before running Qwen, another model or an investigator on post-cutoff factual questions, or picking an investigator for a cutoff task](docs/envs/knowledge_cutoff.md)
- [ACTIVE: Shared Qwen server and RunPod pods, how it works — read when a pod or the shared server misbehaves, or before changing the boot recipe](docs/runpod-qwen-server.md)

## Shipping

Every change reaches `main` through a pull request that a human (Robbie or Kylie) has reviewed.
Never push to `main` directly, and never merge a PR without that human review, even for one-line or
docs-only changes. "Ship" here means: push a branch and open a PR.
