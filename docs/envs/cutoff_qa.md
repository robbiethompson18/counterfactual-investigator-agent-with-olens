# cutoff_qa: factual questions across Qwen's knowledge cutoff

**Status (2026-10-29):** ACTIVE. Bank v2 (803 present-tense questions, 266 stable under Qwen, 187 of
them stable-wrong) is on HF and is the env's default; v1 (1,072 dated questions, 14% stable) is
kept. Finding of the day: asking present tense with no date is what makes the stale trap fire. One
arm (`nohint`); hints are ruled out for this env.

Env: [`environments/cutoff_qa/`](../../environments/cutoff_qa/). Banks:
[`banks/cutoff_qa/`](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens/tree/main/banks/cutoff_qa)
on the sprint HF dataset (`v1.json`, `v2.json`, the v1 subagents' `BRIEF.md`), fetched to
`data/banks/cutoff_qa/` on first use, never in git. Gate code: `scripts/cutoff_gate.py`
([PR #382](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/pull/382));
gate runs under `runs/cutoff_gate/2026-10-29_*` on HF. The current brief for new questions is
[`scripts/cutoff_facts/BRIEF.md`](../../scripts/cutoff_facts/BRIEF.md) (v3, present tense).

## Rules (settled 2026-10-29)

- **The label is Qwen's modal answer (`qwen_modal`), not the truth.** The task is to predict what
  Qwen says; `truth`, `truth_now` and `pre_cutoff_analog` are metadata for reading the results.
- **No numeric answers, anywhere** (Robbie, 2026-10-29): not in any future batch, and the env's
  `items()` drops every item flagged `numeric` (v2: 60, 26 of them stable, so 240 stable items are
  live). Qwen's numbers vary across samples (they fail the gate) and token lenses can't name
  multi-digit values because Qwen splits digits. Never report a numeric stratum; they are out.
- **No date in the question.** The next brief writes present-tense questions directly ("Who is the
  CEO of Intel?"), never "In March 2025, who ...": see the finding below.

## The env

The env asks each question as the only user message, no system prompt, thinking off
(WorkspaceBench's `knowledge_cutoff` render), to Qwen3.6-27B bf16 on the shared vLLM server at
temperature 1.0, top_p 1.0, no presence penalty: WorkspaceBench's `knowledge_cutoff` sampling and
the gate's, so O-lens reads of env transcripts are on text sampled like the gate's. Items are the
stable (≥ 7/8) items of bank v2 by default; `CUTOFF_QA_ALL=1` takes all 803, `CUTOFF_QA_BANK=v1` the
first bank (every episode's meta records `bank`). It labels the reply without an LLM judge:
`correct` (normalized reply equals the truth or an alias), `stale` (equals `qwen_modal`; for v1
items, the pre-cutoff analog), and `contains_truth` / `contains_stale` for replies that add words.
Normalization (`env.norm`): lowercase ASCII, quotes and punctuation stripped (decimal points kept),
leading article dropped, digit-group commas, currency and unit words stripped from numbers ("$32.5
million" → `32.5`), whitespace collapsed.

## Bank v2 (default)

803 present-tense, dateless questions: 629 are rewrites of v1 items that had a natural present-tense
form (Sonnet rewriters; `question_dated` keeps the original, `present_ok` the rewriter's verdict),
174 are new fame-steered questions (brief v2) rewritten the same way (`source_batch` = `v1` / `v2`).
Every item carries the gate's fields from the Qwen runs (`qwen_modal`, `qwen_modal_count`,
`qwen_greedy`, `gate_class` ∈ stable_right / stable_wrong / unstable / refusal_or_null, `gate_run`)
and `numeric`. The 266 stable items were rechecked on the web as of 2026-10-29 (one Sonnet pass:
`truth_now`, `truth_now_source`, `truth_now_as_of`, `truth_changed`, `truth_now_note`).

| v2                                                       | items                               |
| -------------------------------------------------------- | ----------------------------------- |
| total                                                    | 803 (629 from v1, 174 fame-steered) |
| stable (≥ 7/8), right                                    | 79                                  |
| stable, wrong (Qwen's stale answer)                      | 187                                 |
| unstable                                                 | 531                                 |
| refusal or null                                          | 6                                   |
| stable items whose truth changed since writing (recheck) | 13; 3 unverifiable                  |
| stable numeric items (flagged)                           | 26                                  |

### The present-tense finding

The same 174 fame-steered questions, four gate cells (8 samples at T 1 + greedy, thinking off):

| cell                         | run                           | stable_right | stable_wrong | unstable | refusal |
| ---------------------------- | ----------------------------- | ------------ | ------------ | -------- | ------- |
| dated, no system prompt      | `2026-10-29_v2-fame`          | 38 (22%)     | 3 (2%)       | 116      | 17      |
| dated, trivia system prompt  | `2026-10-29_v2-dated-sys`     | 40 (23%)     | 10 (6%)      | 124      | 0       |
| present tense, no system     | `2026-10-29_v2-present-nosys` | 23 (13%)     | 56 (32%)     | 94       | 1       |
| present tense, trivia system | `2026-10-29_v2-present-sys`   | 14 (8%)      | 54 (31%)     | 106      | 0       |

Dated questions name a month after the cutoff, and Qwen either refuses or scatters; the same thing
asked present tense gets a confident stale answer on a third of items (2% → 32% stable-wrong). The
trivia system prompt ("always give a specific answer, never comment on dates") removes the refusals
but does not raise stability. The v1 items rewritten to present tense behave the same way
(`2026-10-29_v1-present`, 629 items): 56 stable-right (9%), 131 stable-wrong (21%), 437 unstable, 5
refusal, against 115 / 32 / 874 / 51 for the full dated v1.

## Bank v1 (dated; `CUTOFF_QA_BANK=v1`)

1,072 dated short-answer questions about events from January 2024 to September 2026 ("Which album
won British Album of the Year at the Brit Awards in February 2026?"), each ending "If you are not
sure, give your best guess. Answer with just the `<answer_type>`." Fields: lowercase canonical
`truth`, `truth_aliases`, `pre_cutoff_analog` for 627 items (58%; the older value a Jan-2025 model
might give), `domain`, `event_date` (`YYYY-MM`), `answer_type`, `source_url`, `verified_by`,
`notes`, `numeric`.

| v1                                      | count                              |
| --------------------------------------- | ---------------------------------- |
| events in 2024 / 2025 / 2026            | 320 / 531 / 221                    |
| with a pre-cutoff analog                | 627                                |
| numeric truth                           | 135                                |
| verified by two agreeing search results | 594 (61% of the 978 batch-2 items) |
| verified on a fetched page              | 384 (39%)                          |
| pilot items, snippet-verified           | 94 (`b1-*`)                        |

### v1 gate (dated, `2026-10-29_b1b2-1072`)

8 samples per question, Qwen3.6-27B bf16 on the shared server, thinking off, T 1.0 / top_p 1.0.
Stable = the same normalized answer in ≥ 7 of 8.

| outcome       | items |
| ------------- | ----- |
| stable, right | 115   |
| stable, wrong | 32    |
| unstable      | 874   |
| refusal       | 51    |

By event year, 2024 questions are 30% stable-right; 2025 and 2026 are 2%. The instability is
genuine, not a normalization artefact: on obscure facts Qwen spreads over many different answers.
Two fixes followed: the fame-steered brief (v2's 174 new items) and, the one that mattered, asking
present tense without a date (above).

### How v1 was made (2026-10-29)

A 100-question pilot (5 Sonnet subagents: business, science, space, sports, tech), then 20 domains ×
50 questions, one Sonnet subagent each (AI, aviation and rail, auctions and records, automotive,
built infrastructure, corporate events, economics and markets, film and TV, football, video games,
health/internet/publishing, music, PC hardware, physical and life sciences, politics, annual
rankings, smartphones, software and standards, spaceflight, other sports). Every agent read the v1
brief (on HF beside the bank) and an avoid-list of every existing question, started from Wikipedia
"2024/2025/2026 in X" pages, list pages and release histories, and verified each answer either on a
page it fetched or by two independent search results agreeing (the pilot accepted one result's
snippet). The brief's rules: one event pinned by month and year, one short unambiguous answer that a
smart reader could not derive from the question, one question per record / office / annual list,
post-cutoff questions preferring a stale trap with the old value recorded as the analog, nothing
after September 2026.

[`scripts/cutoff_facts/merge.py`](../../scripts/cutoff_facts/merge.py) merged the 25 JSONL files
(1,100 rows), checked the schema and dates, flagged questions containing their own truth (0), listed
truths shared by several items (37, e.g. Carlos Alcaraz three times: different events, kept) and
wrote the bank plus a Markdown table. Dedup was manual, by reading the table: 6 pilot rows dropped
in the pilot's own cleaning, then 20 same-event duplicates (the better-phrased one kept) and two
derivable answers (misc-32 "Apple TV", phone-09 "Camera Control") dropped at the merge; the 28 ids
are `DROPPED` in the script, so it reproduces the bank byte for byte. Three pilot rows whose
`domain` field held the question suffix (a subagent bug) were repaired (`FIXES`).

## Known weaknesses

- **Numeric truths** (v1 135, v2 60, 26 of them stable) are flagged `numeric` and dropped by `items()`
  (rule above); dates like "february 2026" and version-ish names like "a19 pro" are not flagged, so
  a few number-like answers stay.
- **Stale analogs are partly from memory.** The v1 brief asked for the analog but not for its
  verification, so many are the subagent's recollection, and 109 are prose ("Bernard Arnault was
  also #1 on the 2023 list") rather than a value. In v2 the gate's `qwen_modal` replaces the analog
  as the stale label, which is the point.
- **Truth drifts.** A present-tense question's answer keeps changing; `truth` is the writer's value
  at authoring time and `truth_now` the 2026-10-29 recheck (13 of 266 stable items had changed, 3
  could not be verified). Re-check before quoting accuracy.
- **Fetched pages were summarizer digests.** The subagents' WebFetch returned a model-written
  summary of the page, not its text, so a "fetched page" verification is one more model reading the
  source. The "two search results" items rest on snippets agreeing.
- 23 v1 `sci-*` questions end with a phrase that differs from their `answer_type` field ("scientific
  name" vs `species name`); harmless for the env, which sends the question as is.
- The pilot's 94 items (`b1-*`) are snippet-verified and use coarser domains than the batch.

## How to generate more

1. Write the avoid-list: every existing question with its answer (one line each) from the current
   bank, so new agents don't ask about the same thing or answer.
2. One Sonnet subagent per domain, 50 questions each, given
   [`scripts/cutoff_facts/BRIEF.md`](../../scripts/cutoff_facts/BRIEF.md) (v3: present tense, no
   date, famous things, half changed since January 2025 and half unchanged, no numbers, the writer's
   predicted stale value as a bare `pre_cutoff_analog`) and the avoid-list, writing one JSONL file
   per agent with the brief's exact keys.
3. `uv run python scripts/cutoff_facts/merge.py --batch DIR/*.jsonl` and read its output: fix
   invalid rows, decide on leaks and shared truths.
4. Manual same-event dedup from the Markdown table; add the dropped ids to `DROPPED`, bump the bank
   version in `merge.py` and `--upload`.
5. Gate the new items with `scripts/cutoff_gate.py` (8 samples at T 1 + greedy, thinking off, no
   system prompt) and attach `qwen_modal` / `gate_class`; recheck the stable items' truth on the web
   and record `truth_now`.
