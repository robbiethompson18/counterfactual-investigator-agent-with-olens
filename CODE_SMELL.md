# Code smells

Tech debt we know about, newest first.

## 2026-10-28: metamodel reader bugs from a Codex review of the serving code

A Codex read of `olens/src/olens/`, `scripts/olens_read_requests.py`,
`scripts/jpp_read_requests.py`, `src/.../render.py` and `runpod/olens_reads.sh` (no GPU run). It
found no layer-index, norm or position-alignment bug in the J++ or O-lens paths. It also found
these (its seed, thinking-render, resume and stale-LoRA bugs are fixed and removed):

- **Field matching can run into a later message** (`render.py` ~78-88): `text.find(s, cur)` searches
  the whole rest of the transcript, so a short field (an assistant's `"A"`) can match inside a later
  message and shift every field after it; an answer `"think"` matches inside `<think>`. Single-turn
  envs are safe; multi-turn ones aren't. Fix: get field spans from the renderer, or match within
  message boundaries.
- Smaller: template tokens carry the previous message's id/role (cfbench places them itself since
  harness v8); `char_offset` ignores a field's stripped leading whitespace (reads land a few chars
  early); per-token decoding mangles characters split across tokens in saved transcript text; J++
  provenance records a blob hash, not the commit, and J++ ignores per-request `layers`.

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

- 2026-10-29: `environments/cutoff_qa/env.py` `measure()` returns `correct`/`stale` but no
  `positive`, so `cfbench.envs.env_runs` (and `scripts/olens_requests_from_runs.py`) skip every
  cutoff_qa row. `cfbench/evals/cutoff_qa_inline` builds cases and the read requests straight from
  the episodes. Give it `positive` (= `stale`?) on PR #380's branch.
