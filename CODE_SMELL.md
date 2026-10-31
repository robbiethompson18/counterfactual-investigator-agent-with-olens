# Code smells

Tech debt we know about, newest first.

## 2026-10-29: `links.json` is built from the local runs only

`spec.build_links` writes `envs/<env>/links.json` (per episode: Docent run, primary value, O-lens
reads; the env viewer's episode chips) from the run folders in this checkout, unlike `env_index`,
which merges HF's index. `environments run` and `upload` ship it, so a checkout without every run of
an env drops the others' chips: HF's hint_mcq `links.json` covers 17 of its 25 runs. It is also why
the hint_mcq census runs' chips still show pre-strict ✓/✗ after PR #416 (a rebuild from a sparse
checkout would drop runs). Index uploads race too: hint_mcq's index had lost four runs
(`2026-10-28_rest*`, `_words`) until PR #416 rebuilt it. Fix: merge with HF's `links.json` the way
`env_index` merges the index.

## 2026-10-29: cfbench.eval `--reuse` keys on the case, not the prompt

The reuse key is (version, item, model, arms, condition, predictor, draw) plus asserts on eval.md /
BASE_LINE / ASK / PROMPT / PLACEBO source, so for the `*_inline` conditions it can't see the O-lens
blocks (data, not code). A new eval with new reads reused another eval's `olens_inline` rows
wholesale (2026-10-29, caught and deleted before scoring). It now also refuses a run with a
different INLINE hook source, which still misses a change in the reads file behind the same hook;
hashing each built prompt into the row would close it.

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

## 2026-10-25: RunPod tooling that cost ~$320 (see runpod/CLAUDE.md, "Cost guards")

- `runpodctl pod get`'s `runtime` is `None` for running pods too (GraphQL showed 674 s uptime for a
  pod runpodctl called runtime-less); the REST `publicIp` + `portMappings["22"]` are the truth.

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
