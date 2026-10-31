# Code smells

Tech debt we know about, newest first.

## 2026-10-31: saved `menus_inline` inputs contain Unicode corruption and miscounted MCQ options

`cfbench/evals/menus_inline/eval.py` reconstructed prompts by joining individually decoded tokens,
which replaces characters split across UTF-8 token boundaries with U+FFFD. Future input preparation
now extracts the user turn from the intact `text` in the fill run's `reads.transcripts.jsonl`,
keeping the token arrays and read positions unchanged (harness 15).

Existing `2026-10-30_menus-inline-inputs/cases.jsonl` inputs and their results remain unchanged:
52/2,523 MMLU-Pro prompts and 51/3,000 SuperGPQA prompts are corrupted, 103/5,523 (1.9%) combined.
The impact on measured lens uplift is unmeasured; the affected fraction does not establish that it
is negligible. Robbie explicitly deferred reruns on 2026-10-31. No saved inputs or results were
regenerated for this fix.

Option parsing also matched lettered lines inside question stems: 7 MMLU-Pro and 33 SuperGPQA cases
have wrong saved option counts, yielding 10 seed-0 placebo pairs with different true option counts.
Two SuperGPQA cases were excluded as apparent 12/14-option singletons; they actually have 8/10
options and can have same-count placebos. Future preparation uses the datasets' structured option
lists, checked against the source evals' saved counts. The same-count placebo guard stays. The
effect on uplift is unmeasured, and Robbie deferred these reruns too; historical inputs and results,
including the two missing cases, remain unchanged.

## 2026-10-31: `honest_score_predict --provider` is a no-op

`scripts/honest_score_predict.py` pins a provider by wrapping `openrouter.chat` in
`functools.partial(..., provider=...)`, but `cfbench.predict.predict` passes `provider=PROVIDER`
(None unless `$CFBENCH_PROVIDER`) explicitly, and a call's keyword beats the partial's, so the pin
never reaches OpenRouter. Use `CFBENCH_PROVIDER='{"order": [...], "allow_fallbacks": false}'`
instead, or drop the flag. (`--google-key` wraps `chat` the same way with `key_env`, which `predict`
doesn't pass, so that one works.)

## 2026-10-30: two MCQ envs build the same prompt with copied code

`environments/hint_mcq` (its `nohint` arm, the MMLU-Pro pool) and `environments/supergpqa` build the
same prompt, each with its own copy of the code: stem, `\n`, the options as `A. …` from `LETTERS`, a
blank line, then `INSTRUCTION = "Answer with the letter only."`. Each also has its own copy of the
bare-letter reply parser (`hint_mcq.BARE_LETTER` vs `supergpqa.LETTER`). The bytes match today
(checked 2026-10-30 on all 2,579 mcq_unselected and 3,000 supergpqa read prompts), and the
mcq_unselected / supergpqa_inline results compare the two pools as if one prompt, so a one-sided
edit would silently break that. Consolidate, details TBD (Robbie, 2026-10-30): probably one shared
`mcq_prompt(stem, options)` + `bare_letter(reply)` that both envs import, keeping two envs since
hint_mcq's ~20 hint arms are a different experiment, not one env with a dataset switch.

## 2026-10-29: `links.json` is built from the local runs only

`spec.build_links` writes `envs/<env>/links.json` (per episode: Docent run, primary value, O-lens
reads; the env viewer's episode chips) from the run folders in this checkout, unlike `env_index`,
which merges HF's index. `environments run` and `upload` ship it, so a checkout without every run of
an env drops the others' chips: HF's hint_mcq `links.json` covers 17 of its 25 runs. It is also why
the hint_mcq census runs' chips still show pre-strict ✓/✗ after PR #416 (a rebuild from a sparse
checkout would drop runs). Index uploads race too: hint_mcq's index had lost four runs
(`2026-10-28_rest*`, `_words`) until PR #416 rebuilt it. Fix: merge with HF's `links.json` the way
`env_index` merges the index.

## 2026-10-29: `cutoff_qa` banks v2 and v3 are not reproducible from the repo

`scripts/cutoff_facts/merge.py` only builds bank v1. v2 and v3 of `banks/cutoff_qa` were assembled
by scratch scripts (clean, dedup, gate-attach, truth recheck) that were not kept. Next batch: port
the clean into `merge.py` so a bank version is reproducible from the agents' JSONL files plus the
gate run.

## 2026-10-29: cfbench.eval `--reuse` keys on the case, not the prompt

The reuse key is (version, item, model, arms, condition, predictor, draw) plus asserts on eval.md /
BASE_LINE / ASK / PROMPT / PLACEBO source, so for the `*_inline` conditions it can't see the O-lens
blocks (data, not code). A new eval with new reads reused another eval's `olens_inline` rows
wholesale (2026-10-29, caught and deleted before scoring). It now also refuses a run with a
different INLINE hook source, which still misses a change in the reads file behind the same hook;
hashing each built prompt into the row would close it.

## 2026-10-28: `environments run` crashes after a finished run in the evals index step

`environments/_lib/run.py` calls `build_index` (`environments/_lib/index.py`), which builds `evals/`
(`build_evals` → `evals.py` → `load_env`) after every env run, before the HF upload, so a failure
there exits 1 after "done: N episodes" without uploading the episodes. An env run shouldn't depend
on every eval's item metadata; catch per eval or skip the evals build there. Ways it has failed:

- **An eval can't import a stored row:** if a cfbench run on disk has rows whose `source` names an
  eval version the local checkout doesn't define (e.g. `mmlu_pro_pause` rows from another branch),
  `ITEM_INFO` raises `KeyError` (three hint_mcq runs on 2026-10-28 had to be uploaded by hand with
  `runfolder.upload`).
- **A missing eval env module (partly fixed, essay_curious PR #306):** `load_env` imported
  `environments.dumb_user_exploit_appraisal`, which does not exist, so every `environments run` on
  main ended with a traceback (`envs/hint_mcq/2026-10-28_late` and `_words` were uploaded by hand
  with `HfApi.upload_folder`). `evals.Envs` now skips envs that don't import.
- **The upload still fails after that fix:** `build_index` also writes `data/evals/*.json`, and
  `runfolder.upload` only accepts paths under `data/envs/`
  (`ValueError: ... evals/... is not in the subpath of .../envs`), so essay_curious's runs were
  uploaded with a direct `upload()` of the run dirs.

HF's top-level `envs/index.json` was down to one env on 2026-10-27: an upload replaces it with
whatever envs the local checkout has, so uploads from sparse checkouts drop the others (restored by
hand for notice_defect).

## 2026-10-28: `cfbench.run --skip` keys on the predictor string, so a route change redoes every draw

`done` in `cfbench/run.py` is keyed on `(env, base_arm, cf_arm, condition, predictor, draw)`. The
same model under another route (`google/gemini-3.8-flash` via OpenRouter, `gemini-direct/…`,
`vertex/…`) counts as a different predictor, so a fill run on a new route repeats every draw the
others already did (on 2026-10-28 arm B ran 228 draws to fill ~60). Workaround: pass skip files with
`predictor` rewritten to the new route. Fix: key on the model name with the route prefix stripped.

## 2026-10-28: `runpod/watch_pod.sh` can't fetch outputs from vLLM-image pods

Its done-marker fetch is `rsync` over ssh, and the `vllm/vllm-openai` image (`v0.27.1`, the read
pods' image) has no rsync, so the fetch fails ("rsync: command not found") and the watcher still
deletes the pod without fetching. Outputs on the network volume survive (the NLA scale2k reads were
pulled over S3), but anything written outside `/workspace` is gone. First seen 2026-10-27: the J++
scale2k read (`jpp_reads/2026-10-27_hint-mcq-scale2k`) was saved by pulling over ssh by hand. Fix:
install rsync in the pod boot (`olens_reads.sh setup`), or fetch with `ssh ... tar c | tar x` (or
`scp`).

## 2026-10-28: metamodel reader bugs from a Codex review of the serving code

A Codex read of `olens/src/olens/`, `scripts/olens_read_requests.py`,
`scripts/jpp_read_requests.py`, `src/.../render.py` and `runpod/olens_reads.sh` (no GPU run). It
found no layer-index, norm or position-alignment bug in the J++ or O-lens paths. It did find these,
none fixed yet:

- **One seed for every lens request** (`one_engine.py` ~203-211, `olens_read_requests.py` ~248):
  vLLM seeds each request's sampler from the same `seed` (default 0), so transcripts that share a
  prompt (e.g. the 8 samples of one item) get identical sampled lens draws on that prompt, not
  independent ones. Fix: seed from (run seed, id, position, layer, replicate).
- **Field matching can run into a later message** (`render.py` ~78-88): `text.find(s, cur)` searches
  the whole rest of the transcript, so a short field (an assistant's `"A"`) can match inside a later
  message and shift every field after it; an answer `"think"` matches inside `<think>`. Single-turn
  envs are safe; multi-turn ones aren't. Fix: get field spans from the renderer, or match within
  message boundaries.
- **Thinking-off generation prompts render with thinking on** (`render.py` ~72): `render_request()`
  doesn't pass `thinking` to `chat_text()`, so an open generation header ends `<think>\n` instead of
  `<think>\n\n</think>\n\n`. Reads at that header see a prompt the model never saw. Transcripts that
  include the reply are unaffected.
- **Resume trusts partial transcripts** (`olens_read_requests.py` ~163, `jpp_read_requests.py` ~51):
  any id with a row in the output counts as done, so a crash mid-transcript leaves it partial
  forever. Fix: a separate done marker, or check the expected position count.
- **Stale converted LoRAs on a reused volume** (`olens_reads.sh` ~42-50): converted adapters are
  rebuilt only when missing, so an updated checkpoint keeps serving the old conversion while
  provenance reports the new revision.
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

## 2026-10-26: old-lens w/ CoT runs still in `cfbench.scoring.groups()`

`2026-10-22_gflash-olens-v2-{exfil,inline-b}` stay in `groups()` only for `--allow-old-lens`; drop
them when the thinking-on reruns land.

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
