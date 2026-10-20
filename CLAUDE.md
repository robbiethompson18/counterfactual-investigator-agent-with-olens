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
- **Metamodel hallucination.** O-lens and NLA readouts invent names, numbers, and plans that aren't
  in the activation. The O-lens gives a per-bullet FVE, which is a cheap hallucination signal to
  pass to the agent.

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
private O-lens research Slack log. Read them there; don't copy them into this repo.

## Data

`data/` (gitignored) mirrors the public HF dataset
[Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens):
O-lens banks in `banks/`, agent runs in `runs/`. Pull with
`hf download Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens --repo-type dataset --local-dir data`;
push new runs back with `hf upload ... data/runs runs --repo-type dataset`. CHIVE incidents
themselves come from
[agu18dec/chive-forensics](https://huggingface.co/datasets/agu18dec/chive-forensics).

Cases viewer: [chive-cases.pages.dev](https://chive-cases.pages.dev) shows all 204 held-out
incidents with every transcript-only and O-lens draw, filterable and linkable by URL.
`viewer/index.html` is the page, `viewer/behavior_kind.tsv` our hand labels; rebuild and redeploy
with `scripts/deploy_viewer.sh` (Cloudflare Pages, needs `CLOUDFLARE_API_TOKEN`).

## Stack

- Python: `uv` for deps/venv, `ruff` for lint + format (line length 140), `ty` for types. Run
  `uv run ruff check . && uv run ruff format . && uv run ty check` before shipping.
- Markdown: Prettier, 100 cols, `proseWrap: always`. Prettier is Markdown-only here.
- Tests: none until this repo is roughly > 50k LOC. Verify by running the code, not by adding a test
  suite. If a test would genuinely save time, ask first.
- Secrets/machine-specific env go in `.envrc.local` (gitignored), never `.envrc`.
- Agam's harness is a git submodule at `vendor/detective-joracle`, installed as an editable uv
  workspace member (`import detective_joracle`). After cloning, run `git submodule update --init`.
  Don't edit it in place; wrap or monkeypatch it from `scripts/`. Ruff and ty skip `vendor/`.
- Agam's scripts take pydra `key=value` args parsed as Python literals: `yesno=false` is the string
  `"false"`, which is truthy. Write `False`/`True`, or omit the arg for the default.
- LLM judge: Claude Sonnet 5.5 (`claude-sonnet-5-5`) for every judge and grader call, with 3
  replicates per answer. Agam's published numbers used Opus 5, so ours aren't directly comparable to
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
  — token-level viewer, CHIVE incidents with Agam's per-token banks, read cache + shared queue,
  SkyPilot pod/fleet setup and its gotchas
- [olens library — read before loading the agu18dec O-lens/AR checkpoints](olens/README.md) —
  hook-LoRA loaders, capture, AR, whitened FVE conventions
- [Smoke25 results — read before running Agam's CHIVE harness with Flash or comparing to his numbers](docs/smoke25-results.md)
  — our wrapper reproduces his free-form pass@1; Flash hits the call cap on yes/no with O-lens
- [Model organisms on Qwen3.6-27B — read before running or reading organism rollouts/O-lens reads](docs/model-organisms.md)
  — which public LoRA organisms work, the rollout and paired organism/base O-lens scripts, first
  findings
- [Robbie's favorite transcripts — read when picking showcase transcripts or "why did you do X?" follow-up targets](docs/favorite-transcripts.md)
  — running list of the most interesting WeirdChat/CHIVE transcripts, with viewer links and ids
- [runpod/ — read before Robbie's agents start any GPU](runpod/CLAUDE.md) — Robbie uses RunPod via
  `runpod/pod.sh`; SkyPilot and bramble GPUs are Kylie's

## Shipping

Every change reaches `main` through a pull request that a human (Robbie or Kylie) has reviewed.
Never push to `main` directly, and never merge a PR without that human review, even for one-line or
docs-only changes. "Ship" here means: push a branch and open a PR.
