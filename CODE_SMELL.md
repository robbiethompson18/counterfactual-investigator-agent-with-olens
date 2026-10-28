# Code smells

Tech debt we know about, newest first.

## 2026-10-28: callers of the deleted `cfbench.envs.env_runs.rows()`

`rows()` (episodes as evalcue-shaped rows) went in `b78d8bd` (2026-10-27) but two callers remain,
silenced with `ty: ignore` in the ty-clean PR: `environments/hint_mcq/report.py` (so it and
`gate.py`, which imports it, fail at import) and the `envs:` specs of
`scripts/olens_requests_from_runs.py`. Port them to `EnvRuns` / `read_episodes`, or restore
`rows()`.

## 2026-10-27: cfbench places every chat-template token at one character, labelled as the previous message

The readers give template tokens (`<|im_end|>`, `\n`, `<|im_start|>`, `assistant`, `<think>`,
`</think>`, the `\n\n` before the reply) `field: null` and the `msg_index`/`role` of the message
before them, and `cfbench/olens.py` `positions()` puts all of them at that message's end. On
hint_mcq the 9 tokens from the user's `<|im_end|>` to the `\n\n` right before the answer letter all
sit at one char, labelled `user`, with nothing saying which token each read is; a `search_olens`
over the transcript returns only the first of them (the "first read after the match"). The reads are
all there (`olens_at` with the default window and `over="bullets"` reach them), so O-lens and
placebo are handicapped equally, but the read right before the answer is the one that matters on the
answer task.

## 2026-10-27: a request row has one position mode, so header reads are a second `|header` request

`scripts/olens_read_requests.py` reads a row either at stop tokens or at `turn_starts: N` (an
`<|im_start|>` and the N tokens after it), not both. `olens_reads/2026-10-27_hint-mcq-header` and
`-scale2k` got both by submitting each transcript twice, the second with `|header` appended to its
id, and then needed a derived `cfbench/wrong_mcq/` dir that merges the header rows back under the
plain id so they join rollout sources. Fix: let `turn_starts` add to the stop positions in one row
(or have `olens.Reads` strip a `|suffix`).

## 2026-10-27: `runpod/watch_pod.sh` can't fetch from vLLM-image pods

It fetches outputs with `rsync` before deleting the pod, but `vllm/vllm-openai:v0.27.1` (the read
pods' image) has no `rsync`, so on those pods the watcher deletes without fetching. The J++ scale2k
read (`jpp_reads/2026-10-27_hint-mcq-scale2k`) was saved by pulling over ssh by hand. Fix: fall back
to `ssh cat`/`scp`, or install rsync in `olens_reads.sh setup`.

## 2026-10-26: old-lens w/ CoT runs still in `cfbench.scoring.groups()`

`2026-10-22_gflash-olens-v2-{exfil,inline-b}` stay in `groups()` only for `--allow-old-lens`; drop
them when the thinking-on reruns land.

## 2026-10-25: RunPod tooling that cost ~$320 (see runpod/CLAUDE.md, "Cost guards")

- `runpodctl pod get`'s `runtime` is `None` for running pods too (GraphQL showed 674 s uptime for a
  pod runpodctl called runtime-less); the REST `publicIp` + `portMappings["22"]` are the truth.

## 2026-10-22: serving code left outside the shared-server recipe

Left alone by the dead-serving-code cleanup, either because it does something the shared server
can't or because it isn't ours to change:

- (Resolved 2026-10-26: all three deleted, last at commit `7296169`.)
  `scripts/party_resample.py gen` and `transcript-viewer/scripts/thinking.py gen` ran vLLM
  in-process (`LLM(...)`) from token-id prompts; `scripts/identity/pod_setup.sh` set
  `HF_HOME=/workspace/hf`, which `runpod/CLAUDE.md` says not to do.

## 2026-10-21: runs from before `scripts/runs.py` aren't in the layout or on HF

Backfill them into `runs/<experiment>/<date>_<slug>/` with a best-effort `meta.json`
(`"backfilled": true`) once the agents producing them finish:

- robbie-box `~/envs-results/` (agent-interp-envs screens): funding*email, sandbagging (screen and
  iteration), grade_your_own and eval_tampering are backfilled (2026-10-22) as
  `runs/<env>/<date>*{screen,iter}/`. Still only on the box: `optimize*cache_c`(2.2 GB, retired env,
  skipped on purpose), the`plumbing\*`and`fe_smoke*\*`smoke tests, and the gateway logs
  in`~/envs-logs/`. Step dirs are root-owned (Docker), so copy with `sudo`. Their
  `config.yaml`says`provider: fireworks`even when`scripts/envs_gateway.py` routed the calls
  elsewhere, so don't trust it for which server answered.
- Robbie's Mac `data/runs/` (as of 2026-10-22): `identity` and `evalcue` (pre-`runs.py`
  `<env>/TAG.jsonl` files, no `meta.json`) are fully on HF and `cfbench` nearly (7 files
  local-only); `stoptokens` and `vardania_sneeze` are not on HF at all. Of the 13 loose
  `*.log`/`*.txt` files at the root, 4 are local-only (`dev96_yesno_haiku55_retry{1,2}.log`,
  `or_usage_before_*.txt`). `vardania_sneeze/` holds `.py` scripts that belong in git.
- RunPod pods hold no results (they're bare vLLM servers), but nothing records which pod, vLLM
  version or Qwen revision served a run.
