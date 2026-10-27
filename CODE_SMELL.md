# Code smells

Tech debt we know about, newest first.

## 2026-10-26: expview runs that only ever lived on bramble

expview became a static page reading HF (PR "viewers/expview-static"). Three kinds of dataset
linked from docs and scripts aren't on HF, so their `#d=…` links open "no dataset" until someone
uploads the run under `runs/<experiment>/<run>/expview` and `runs.py finish`es it:

- bramble's 27 pre-monorepo runs (`expview/character_drift_*`, `expview/overeager_*`,
  `expview/realism_evalness_*`; docs/envs/character_drift.md, docs/glm53-realism.md,
  `donehack/realism/overeager/*.py`);
- `imports/celeste/…` and `imports/olens_reads/…` views (Celeste's parquet re-tokenised;
  `expview/importers/celeste.py`, `scripts/glm_olens_dense/CLAUDE.md`). The two `olens_prompt_ea`
  overlays on HF point at them, so on the page they show judge calls without the transcript;
- `olens_prompt_ea/*.py` still emit `bramble:8821` (`olens_search`, deleted) links in Record.

Also dropped with the server: Docent links resolved by transcript fingerprint (the page only
resolves `extra.source` / `run_dir` + `row_index` through `docent.json`), and "annotations from other
runs" on a base item (needs a cross-run index).

## 2026-10-26: OpenRouter calls outside `openrouter.chat`

`olens_prompt_ea/judge.py` and `judge_punct.py` still hand-roll httpx calls to OpenRouter: they are
standalone PEP 723 scripts (`uv run --script`, Kylie's), so they can't import the package without
pulling the whole project into their script env.

## 2026-10-26: cfbench/report/data.py still computes the log skill

`cfbench.report.data` (RMSE) is the standard. `cfbench/score.py` stopped printing the old binomial
log skill on 2026-10-26 (`cfbench/sweep.py`, log skill only, was deleted the same day);
`data.log_skill` still feeds `summary.json` so numbers from before 2026-10-26 can be checked. Drop
it once nothing quotes them. The old-lens w/ CoT runs
(`2026-10-22_gflash-olens-v2-{exfil,inline-b}`) stay in `groups()` only for `--allow-old-lens`; drop
them when the thinking-on reruns land.

## 2026-10-25: RunPod tooling that cost ~$320 (see runpod/CLAUDE.md, "Cost guards")

- `runpod/pod.sh ssh` has no `ConnectTimeout`, so a pod whose proxy accepts TCP before sshd is up
  hangs the caller forever (a 40-try poll loop sat for an hour).
- `runpod/pod.sh`'s `pod_id`/`ssh_args` call `runpodctl pod get` with whatever `pod_id` printed;
  when the name isn't found that is an empty id, and `runpodctl pod get ''` waits on stdin.
- `runpodctl pod get`'s `runtime` is `None` for running pods too (GraphQL showed 674 s uptime for a
  pod runpodctl called runtime-less); the REST `publicIp` + `portMappings["22"]` are the truth.
- `runpod/qwen_server_boot.sh` starts with `exec > /workspace/boot.log`, so if the network volume
  isn't mounted the pod dies silently with no sshd; log to `/tmp` first and check the mount.
- The O-lens read orchestration (shard launch, resume, fetch, teardown, balance floor) lived in
  session scratchpads, so nothing durable guards the pods; `olens/stream-reads-to-hf` (PR pending)
  is the start of fixing that.
- `scripts/olens_read_requests.py` crashes on a tool call whose `arguments` aren't valid JSON (4
  thinking-off Qwen transcripts); fix in the same PR.

## 2026-10-23: scripts/anchor_reads.py is slow on long transcripts

Resolved 2026-10-26 by deleting the script (last at commit `7296169`) with the null `anchors`
condition; the notes stay for anyone who revives it.

~4.6k tokens/s on an H200 (94 s for a 3.4k-token, 110-sentence transcript). Causes: attention is
explicit fp32 in Python query blocks (needed for the per-row suppression mask and receiver sums; a
flex_attention mask would do both faster), every suppression batch recomputes the unsuppressed row,
rows don't share the prefix before their span (a cache copied at the span start would), and
`causal_conv1d` runs its PyTorch fallback. Fix before reading 10k+ token CoTs.

## 2026-10-23: evalcue drops `top_k` on the OpenRouter path

`evalcue/harness.py` `chat()` replaces `extra_body` wholesale when `EVALCUE_MODEL` isn't `qwen`, so
OpenRouter runs never send `top_k=20` (nor `chat_template_kwargs`). OpenRouter Qwen runs therefore
don't use Qwen's full recommended sampling in either thinking mode, which may be part of why their
Lynch rates sit well below the bf16 vLLM ones ([thinking-off.md](docs/thinking-off.md)). Fixing it
changes the config of every existing OpenRouter run, so it's left alone for now.

## 2026-10-23: evolve/ sandbox and seed caveats (resolved 2026-10-26: `evolve/` deleted)

`evolve/` was Kylie's env-evolution loop (2026-10-23): a baseline env plus a REAL edit that lowers
how often Qwen3.6-27B shows a behaviour, a NULL edit that looks like it should but doesn't, and a
grader; Sonnet "Evolver"/"Creator" agents mutated nodes in podman sandboxes, a Flash forecaster,
auditor and innocuousness judge scored them, and Pareto layers over (L, U, N, I) picked the next
round. Deleted 2026-10-26: nothing in the repo referenced it, it hard-coded `~/workspace/*` paths,
its seeds were never in git, and its spec lives only in a Claude Doc. The code is at commit
`7296169`:
https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/tree/7296169/evolve.
The caveats below are kept for the record.

- `evolve/` sandboxes ran commands as uid 1000 with an agent-owned tmpfs work dir; evalcue ran as
  root with a bind mount. Read-only files could be deleted and replaced; they were restored before
  grading and reported in `record["readonly_tampered"]`, not prevented.
- Podman `--userns auto` gives bramble ~64 sandbox slots shared by every process; `EVOLVE_MAX_BOXES`
  (default 16) was per process, so several concurrent `evolve` processes could still queue on slots.
- Seed nodes (`evolve/seeds/`) were data and not in git, rebuilt by
  `evolve/scripts/convert_seeds*.py`. They (evolve-v1 tip `465c519`, 670 files) and the 23 Oct run
  data (48,264 files) are on the HF dataset under `runs/evolve/`; the 391 Evolver agent transcripts
  were deleted, not uploaded (they contained Claude Code's injected user context) and are listed in
  `runs/evolve/data/WITHHELD_agent_transcripts.txt`.
- The Max-plan job queue (`evolve/job_queue.py`) let subagents work on the host filesystem, not in
  the podman sandbox the API agent used; the manager's instructions were the only guard.

## 2026-10-22: repository-wide type-check baseline is not clean

At `8d6fde7`, `uv run ty check` reports 286 diagnostics outside the new overnight package. The
overnight work adds none; its scoped check passes. Full-project type-check success remains a
separate cleanup task, rather than silently weakening the checker for this experiment.

## 2026-10-22: overnight fixture logs are writable and reveal the simulation

`evalcue/envs/overnight/` uses readable local scripts and an agent-writable JSONL log. Models can
recognize canned replies or alter instrumentation. The initial screens retain this setup for
comparability; do not treat their detector as an adversarially secure grader or rates as deployment
estimates. See [the write-up](docs/envs/overnight.md) for transcript evidence and detection limits.

## 2026-10-22: serving code left outside the shared-server recipe

Left alone by the dead-serving-code cleanup, either because it does something the shared server
can't or because it isn't ours to change:

- (Resolved 2026-10-26: all three deleted, last at commit `7296169`.)
  `scripts/party_resample.py gen` and `transcript-viewer/scripts/thinking.py gen` ran vLLM
  in-process (`LLM(...)`) from token-id prompts; `scripts/identity/pod_setup.sh` set
  `HF_HOME=/workspace/hf`, which `runpod/CLAUDE.md` says not to do.
- Kylie's `weirdness_elo/judge_qwen.py` and `qwen_tourney.py` hardcode `localhost:8000` and the FP8
  checkpoint.

## 2026-10-22: `shutdown_*` doc tables come from a scratch script

The tables in `docs/envs/shutdown-arms.md` were generated by a throwaway script (pool all run dirs,
rescore, Wilson CI) that isn't in the repo; regenerating them means rewriting it. Also,
`shutdown_killswitch`/`shutdown_cron` handle stacked arms with long `if arm in (...)` chains: exact
matches, so safe, but each new stacked arm means editing several tuples.

## 2026-10-22: `donehack/` runs don't go through `scripts/runs.py`

`donehack/port.py` and `variants.py` write to `donehack/runs/<out>/` (gitignored) with no
`meta.json`. The runs behind `docs/donehack-endings.md` were uploaded by hand to the HF dataset
`Robbie-Kylie-NN-sprint/donehack`, not in the `runs/<experiment>/<date>_<slug>/` layout. The bwrap
backend hard-codes NixOS store paths and is kept only for comparison with those runs.

## 2026-10-21: `cfbench/envs/secret_number.py` runs on rows from an unmerged branch's code (resolved: the split, hopeless and pressure arms are in `evalcue/envs/secret_number/__init__.py` on main)

Resolved 2026-10-26 by deletion: the adapter and `evalcue/envs/secret_number/` are gone (last at
commit `7296169`); the rows stay on HF with no code on main behind them.

Its rows (10 `qwen2_*` files in `data/runs/evalcue/secret_number/`, on HF under the same path) were
generated on branch `evalcue-secret-number`, whose newer arms (split, salient, prompt cues) aren't
on main. Arm edits are diffed from the rows, so nothing breaks, but the rows have no provenance link
to code on main until that branch merges. The behavior text is the `Hack` section of main's
`evalcue/envs/secret_number/` docstring, which omits the detector's `tamper` rule (0 tamper-only
hacks in these rows).

## 2026-10-21: runs from before `scripts/runs.py` aren't in the layout or on HF

Backfill them into `runs/<experiment>/<date>_<slug>/` with a best-effort `meta.json`
(`"backfilled": true`) once the agents producing them finish:

- robbie-box `~/envs-results/` (agent-interp-envs screens): funding*email, sandbagging (screen and
  iteration), grade_your_own and eval_tampering are backfilled (2026-10-22) as
  `runs/<env>/<date>*{screen,iter}/`. Still only on the box: `optimize*cache_c`(2.2 GB, retired env, skipped on purpose), the`plumbing\*`and`fe_smoke*\*`smoke tests, and the gateway logs in`~/envs-logs/`. Step dirs are root-owned (Docker), so copy with `sudo`. Their `config.yaml`says`provider:
  fireworks`even when`scripts/envs_gateway.py` routed the calls elsewhere, so don't trust it for
  which server answered.
- Robbie's Mac `data/runs/` (as of 2026-10-22): `identity` and `evalcue` (pre-`runs.py`
  `<env>/TAG.jsonl` files, no `meta.json`) are fully on HF and `cfbench` nearly (7 files
  local-only); `stoptokens` and `vardania_sneeze` are not on HF at all. Of the 13 loose
  `*.log`/`*.txt` files at the root, 4 are local-only (`dev96_yesno_haiku55_retry{1,2}.log`,
  `or_usage_before_*.txt`). `vardania_sneeze/` holds `.py` scripts that belong in git.
- RunPod pods hold no results (they're bare vLLM servers), but nothing records which pod, vLLM
  version or Qwen revision served a run.

## 2026-10-21: main doesn't pass its own `ty` check

`uv run ty check` reports 286 diagnostics (2026-10-22): 98 unresolved imports (torch, transformers,
vllm, `server`, `olens.*`) in GPU-only scripts whose deps live only in the pod venvs, and 152 in
`transcript-viewer/web/server.py`, mostly untyped JSON dicts. Check only the files you touched until
someone excludes the pod-only modules and types `server.py`.

## 2026-10-21: `evalcue/` harness rough edges

- The sandbox runs the agent as root in the container.
- Several envs' `detect()` count wrong-but-honest code that happens to pass the impossible test as a
  hack (`lcb_mock_entity`), or count incomplete work as gaming (`precommit_scope`). See
  `docs/eval-cue-reward-hacking.md`. (Both deleted 2026-10-26; the doc keeps the caveat.)
