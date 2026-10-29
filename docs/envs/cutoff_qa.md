# cutoff_qa: present-tense factual questions across Qwen's knowledge cutoff

**Status (2026-10-29):** ACTIVE. Bank v3 is the env's default: 5,467 dateless present-tense
questions ("Who is the CEO of Intel?"), 2,196 of which Qwen3.6-27B answers stably, 649 of those with
a stale answer. Results so far are in [cutoff-qa-inline.md](../cutoff-qa-inline.md). One arm
(`nohint`); hints are ruled out.

- Env: [`environments/cutoff_qa/`](../../environments/cutoff_qa/). Banks on HF under
  [`banks/cutoff_qa/`](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens/tree/main/banks/cutoff_qa)
  (`v3.json`, `v3_stable.md` readable table, `BRIEF_v4.md`, `lists/` with the 100 entity lists;
  older `v1.json`, `v2.json`), fetched to `data/banks/cutoff_qa/` on first use, never in git.
- Gate: `scripts/cutoff_gate.py`; runs under `runs/cutoff_gate/2026-10-29_*` on HF.
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
stable items of v3; `CUTOFF_QA_ALL=1` takes every item, `CUTOFF_QA_BANK=v2` / `v1` the older banks.
`measure()` labels the reply with no LLM judge: `correct` (normalized reply equals `truth` or an
alias), `stale` (equals `qwen_modal`), `contains_truth` / `contains_stale`. Normalization
(`env.norm`): lowercase ASCII, punctuation and leading article stripped, currency and unit words
stripped from numbers. Each item carries `qwen_modal`, `qwen_modal_count`, `qwen_greedy`,
`gate_class` (stable_right / stable_wrong / unstable / refusal_or_null), `gate_run`, `category`,
`source_batch`, `numeric`, `truth_now` fields, and for rewritten items `question_dated`.

## Bank v3

|                                             | items                                                   |
| ------------------------------------------- | ------------------------------------------------------- |
| total                                       | 5,467 (v2's 803 plus 4,664 from 100 entity-list agents) |
| stable, right (Qwen's answer is still true) | 1,547                                                   |
| stable, wrong (Qwen's answer is stale)      | 649                                                     |
| unstable                                    | 3,258                                                   |
| refusal or null                             | 13                                                      |

Yield by category for the 4,664 entity-list items (stable / written, stale among them): companies
618 / 1,208 (109), countries 440 / 1,206 (131), sports 297 / 736 (105), products 172 / 343 (25),
media 157 / 382 (39), institutions 131 / 334 (29), awards 68 / 355 (20), science 47 / 100 (4).
Annual-award lists (Grammys, Hugo, Goldman, Game Awards: 0 to 7 stable of 50) were a waste;
household-name companies, flagship products and US major-league sports gave 32 to 34 of 50.

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

## Known weaknesses

- **Truth drifts.** `truth` is the writer's value at authoring; present-tense answers keep changing.
  Re-check before quoting accuracy against truth. The label (`qwen_modal`) does not drift.
- **Stale analogs are guesses.** `pre_cutoff_analog` is the writer's prediction; the gate's
  `qwen_modal` is the real stale value.
- **Fetched pages were summarizer digests**, not page text, so a "fetched page" verification is one
  more model reading the source.
- **The agents padded to 50.** Lists of small countries or single-office bodies ran out of famous
  things and filled with speakers, deputy ministers and central-bank governors, which the gate
  mostly rejects. Better a short file.
- No text-inversion gate was applied (it selects on the investigator failing; see CLAUDE.md), so
  transcript-only baselines are honest but high on `stable_right` items.

## How to generate more

1. Write disjoint entity lists (7 to 16 famous entities each with a "what to ask" paragraph; the 100
   used are on HF under `banks/cutoff_qa/lists/`). Skip annual-award categories. Prefer entities
   whose value changed after January 2025 if stale items are the goal.
2. One Sonnet subagent per list, at most 50 concurrent, each given
   [`scripts/cutoff_facts/BRIEF.md`](../../scripts/cutoff_facts/BRIEF.md) plus an avoid-list of
   every existing question. WebFetch the Wikipedia page first; the session's WebSearch budget (1,000
   calls shared by all subagents, `CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION`) drains in minutes
   otherwise.
3. Merge and clean (`scripts/cutoff_facts/merge.py` for the checks: dated, numeric, answer in
   question, duplicates), gate with `scripts/cutoff_gate.py`, attach the gate fields, bump the bank
   version, upload to HF, point the env at it.
