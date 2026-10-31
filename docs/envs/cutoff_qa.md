# cutoff_qa: present-tense factual questions across Qwen's knowledge cutoff

**Status (2026-10-29):** ACTIVE. Bank v4 is the env's default: 6,411 dateless present-tense
questions ("Who is the CEO of Intel?"), 2,605 of which Qwen3.6-27B answers stably, 861 of those with
a stale answer. Results so far are in [cutoff-qa-inline.md](../cutoff-qa-inline.md). One arm
(`nohint`); hints are ruled out.

- Env: [`environments/cutoff_qa/`](../../environments/cutoff_qa/). Banks on HF under
  [`banks/cutoff_qa/`](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens/tree/main/banks/cutoff_qa)
  (`v4.json`, `v4_stable.md` readable table, `BRIEF_v5.md` and `lists_v5/`, `BRIEF_v4.md` and
  `lists/`, the 100 entity lists of each round; older `v3.json`, `v2.json`, `v1.json`), fetched to
  `data/banks/cutoff_qa/` on first use, never in git.
- Gate: `scripts/cutoff_gate.py`; runs under `runs/cutoff_gate/2026-10-29_*` on HF.
- Cleaning of a writer batch: the checks in `scripts/cutoff_facts/merge.py` (schema, closing
  sentence, dated stems, numeric truths, truth in the question, duplicate stems, stems already in
  the bank).
- Authoring brief for new questions:
  [`scripts/cutoff_facts/BRIEF.md`](../../scripts/cutoff_facts/BRIEF.md).

## Rules

- **The label is Qwen's modal answer (`qwen_modal`), not the truth.** The task is to predict what
  Qwen says. `truth`, `truth_now` and `pre_cutoff_analog` are metadata.
- **No date in the question.** "In March 2025, who became CEO of Intel?" makes Qwen hedge or refuse;
  "Who is the CEO of Intel?" gets its stale belief 7 of 8 times. Dated questions were 2%
  stable-wrong, the same questions present-tense 32% (table below).
- **No numeric answers.** Qwen's numbers vary across samples and token lenses can't name multi-digit
  values. v2 still has 26 stable numeric items, flagged `numeric`; exclude them.
- **Stable means 7 of 8 samples agree** (temperature 1, top-p 1, thinking off, no system prompt, the
  question as the only user message). `qwen_modal_count` is kept so a softer bar can be used.

## The env

Asks each question verbatim as the only user message, no system prompt, thinking off, to Qwen3.6-27B
bf16 on the shared vLLM server at temperature 1.0 / top_p 1.0: WorkspaceBench's `knowledge_cutoff`
render and the gate's sampling, so O-lens reads of env transcripts match the gate. `items()` is the
stable items of v4; `CUTOFF_QA_ALL=1` takes every item, `CUTOFF_QA_BANK=v3` / `v2` / `v1` the older
banks. `measure()` labels the reply with no LLM judge: `correct` (normalized reply equals `truth` or
an alias), `stale` (equals `qwen_modal`), `contains_truth` / `contains_stale`. Normalization
(`env.norm`): lowercase ASCII, punctuation and leading article stripped, currency and unit words
stripped from numbers. Each item carries `qwen_modal`, `qwen_modal_count`, `qwen_greedy`,
`gate_class` (stable_right / stable_wrong / unstable / refusal_or_null), `gate_run`, `category`,
`source_batch`, `numeric`, `truth_now` fields, and for rewritten items `question_dated`.

## Bank v4

|                                             | v4 total | v2 (803) | v4-entities (4,664) | v5-changed (944) |
| ------------------------------------------- | -------- | -------- | ------------------- | ---------------- |
| stable, right (Qwen's answer is still true) | 1,744    | 79       | 1,468               | 197              |
| stable, wrong (Qwen's answer is stale)      | 861      | 187      | 462                 | 212              |
| unstable                                    | 3,785    | 531      | 2,727               | 527              |
| refusal or null                             | 21       | 6        | 7                   | 8                |
| stale share of written                      | 13%      | 23%      | 10%                 | 22%              |
| stale share of stable                       | 33%      | 70%      | 24%                 | 52%              |

v3 was v2 plus v4-entities (5,467 items, 2,196 stable, 649 stale); v4 adds v5-changed.

Yield by category for the 4,664 v4-entities items (stable / written, stale among them): companies
618 / 1,208 (109), countries 440 / 1,206 (131), sports 297 / 736 (105), products 172 / 343 (25),
media 157 / 382 (39), institutions 131 / 334 (29), awards 68 / 355 (20), science 47 / 100 (4).
Annual-award lists (Grammys, Hugo, Goldman, Game Awards: 0 to 7 stable of 50) were a waste;
household-name companies, flagship products and US major-league sports gave 32 to 34 of 50.

For the 944 v5-changed items (written / stable / stale): companies 182 / 91 / 26, products 124 / 43
/ 20, offices 120 / 59 / 37, media 111 / 30 / 10, sports 105 / 65 / 53, subnational 78 / 41 / 27,
culture 73 / 25 / 17, institutions 62 / 17 / 8, memberships 45 / 28 / 9, books 44 / 10 / 5. Sports
(coaches, champions, transfers) is the richest stale vein: half its stable items are stale.

## How it was made (all 2026-10-29)

1. **v1, 1,072 dated questions** from 25 Sonnet subagents (one domain each) with web search: 14%
   stable. Instability was genuine, not normalization: on obscure facts Qwen spreads over many
   answers.
2. **v2, 803 present-tense questions:** the 629 v1 items with a natural present-tense form,
   rewritten by Sonnet, plus 174 new fame-steered ones. 266 stable, 187 stale. Truth rechecked on
   the web for the stable items (13 had changed since authoring, 3 unverifiable).
3. **v3, plus 4,664 from 100 Sonnet subagents**, each given a disjoint list of famous entities (25
   country lists, 25 company, 15 sports, 8 each awards / products / institutions / media, 3 science)
   and brief v4: fetch the entity's Wikipedia page, present tense, no dates, no numbers. 4,862
   written; dated, numeric, answer-in-question and duplicate items dropped. Every answer was
   verified on a page fetched the same day, so `truth_now` = `truth`.
4. **v4, plus 944 from 100 more Sonnet subagents with changed-facts-first lists.** Ten Sonnet
   drafters first wrote 100 disjoint lists of famous entities with a note on what changed about each
   after January 2025 (`lists_v5/`); the writers (brief v5) verified each claim on the fetched page,
   asked about the changed facts first and stopped when the list ran dry (median 10 questions).
   1,078 written in two waves of 50 agents, 944 kept (119 repeated a v3 question, 10 leaked the
   answer, 3 dated, 2 duplicates). The stale share of stable items doubled (24% → 52%).

Gate runs, Qwen3.6-27B, 8 samples plus greedy:

| run (`runs/cutoff_gate/2026-10-29_*`) | items | question form                      | stable right | stable wrong | unstable | refusal |
| ------------------------------------- | ----- | ---------------------------------- | ------------ | ------------ | -------- | ------- |
| `b1b2-1072`                           | 1,072 | v1, dated                          | 115          | 32           | 874      | 51      |
| `v2-fame`                             | 174   | fame-steered, dated                | 38           | 3            | 116      | 17      |
| `v2-dated-sys`                        | 174   | same, "never refuse" system prompt | 40           | 10           | 124      | 0       |
| `v2-present-nosys`                    | 174   | same, present tense                | 23           | 56           | 94       | 1       |
| `v2-present-sys`                      | 174   | present tense + system prompt      | 14           | 54           | 106      | 0       |
| `v1-present`                          | 629   | v1 rewritten present tense         | 56           | 131          | 437      | 5       |
| `v4-entities`, `v4-entities-e087b`    | 4,664 | entity lists, present tense        | 1,468        | 462          | 2,727    | 7       |
| `v5-changed`                          | 944   | changed-facts-first lists          | 197          | 212          | 527      | 8       |

## Known weaknesses

- **Banks v2 and v3 can't be rebuilt from the repo**: `scripts/cutoff_facts/merge.py` builds only v1,
  and the scratch scripts that cleaned, deduplicated and gate-attached v2 and v3 weren't kept; the
  banks themselves are on HF.

- **Truth drifts.** `truth` is the writer's value at authoring; present-tense answers keep changing.
  Re-check before quoting accuracy against truth. The label (`qwen_modal`) does not drift.
- **Stale analogs are guesses.** `pre_cutoff_analog` is the writer's prediction; the gate's
  `qwen_modal` is the real stale value.
- **Fetched pages were summarizer digests**, not page text, so a "fetched page" verification is one
  more model reading the source.
- **The v4-entities agents padded to 50.** Lists of small countries or single-office bodies ran out
  of famous things and filled with speakers, deputy ministers and central-bank governors, which the
  gate mostly rejects. The v5 brief says to stop instead; the v5 agents averaged 11 questions.
- **The avoid-list eats the famous changes.** By round v5 most famous changed offices were already
  in the bank, so writers returned a median 10 questions and 119 of 1,078 repeated a v3 stem anyway
  (dropped). A further round needs new entity kinds, not more of the same lists.
- **Some `truth` fields in v3 are stale** (the v5 writers noticed about eight: the Taoiseach, the
  French prime minister, the Nepal and Ukraine prime ministers, the FDA commissioner, the White
  House press secretary, the Vuelta and snooker winners, the VMA Video of the Year). The label
  `qwen_modal` is unaffected; `correct` against those items is not.

## How to generate more

1. Write disjoint entity lists (7 to 16 famous entities each, with what changed about each after
   January 2025 and what to ask; the 200 used are on HF under `banks/cutoff_qa/lists/` and
   `lists_v5/`). Skip annual-award categories. Changed-facts-first lists (`lists_v5/`, written by 10
   Sonnet drafters) doubled the stale share of stable items; pick entity kinds the bank doesn't
   already cover, since writers mostly return what the avoid-list doesn't already have.
2. One Sonnet subagent per list, at most 50 concurrent, each given
   [`scripts/cutoff_facts/BRIEF.md`](../../scripts/cutoff_facts/BRIEF.md) plus an avoid-list of
   every existing question. WebFetch the Wikipedia page first; the session's WebSearch budget (1,000
   calls shared by all subagents, `CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION`) drains in minutes
   otherwise.
3. Merge and clean (`scripts/cutoff_facts/merge.py` for the checks: dated, numeric, answer in
   question, duplicates, stem already in the bank), gate with `scripts/cutoff_gate.py` (944 items
   took 4 minutes on the shared server at concurrency 32), attach the gate fields (`qwen_modal`,
   `qwen_modal_count`, `qwen_greedy`, `gate_class`, `gate_run`, `truth_now` = `truth`), bump the
   bank version, upload to HF, point the env at it.
