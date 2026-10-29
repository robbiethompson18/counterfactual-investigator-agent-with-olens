# cutoff_qa: factual questions across Qwen's knowledge cutoff

**Status (2026-10-29):** ACTIVE. Bank v3 (5,467 present-tense questions: v2's 803 plus 4,664 from
100 entity-list agents; 2,196 stable under Qwen, 649 of them stable-wrong) is on HF and is the env's
default; v2 (803, 266 stable) and v1 (1,072 dated, 14% stable) are kept. Finding of the day: asking
present tense with no date is what makes the stale trap fire; assigning entities instead of domains
is what scales it. One arm (`nohint`); hints are ruled out for this env.

Env: [`environments/cutoff_qa/`](../../environments/cutoff_qa/). Banks:
[`banks/cutoff_qa/`](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens/tree/main/banks/cutoff_qa)
on the sprint HF dataset (`v1.json`, `v2.json`, `v3.json`, their `*_stable.md` tables, the v1
subagents' `BRIEF.md`, `BRIEF_v4.md`, and `lists/` with the 100 entity lists), fetched to
`data/banks/cutoff_qa/` on first use, never in git. Gate code: `scripts/cutoff_gate.py`
([PR #382](https://github.com/robbiethompson18/counterfactual-investigator-agent-with-olens/pull/382));
gate runs under `runs/cutoff_gate/2026-10-29_*` on HF. The current brief for new questions is
[`scripts/cutoff_facts/BRIEF.md`](../../scripts/cutoff_facts/BRIEF.md) (v4: entity lists, present
tense); how the lists were built is in
[`scripts/cutoff_facts/entity_lists.md`](../../scripts/cutoff_facts/entity_lists.md).

## Rules (settled 2026-10-29)

- **The label is Qwen's modal answer (`qwen_modal`), not the truth.** The task is to predict what
  Qwen says; `truth`, `truth_now` and `pre_cutoff_analog` are metadata for reading the results.
- **No numeric answers** in any future batch: Qwen's numbers vary across samples (they fail the
  gate) and token lenses can't name multi-digit values because Qwen splits digits. v2 still has 26
  stable numeric items (60 in all), flagged `numeric`; stratify or exclude them.
- **No date in the question.** The next brief writes present-tense questions directly ("Who is the
  CEO of Intel?"), never "In March 2025, who ...": see the finding below.

## The env

The env asks each question as the only user message, no system prompt, thinking off
(WorkspaceBench's `knowledge_cutoff` render), to Qwen3.6-27B bf16 on the shared vLLM server at
temperature 1.0, top_p 1.0, no presence penalty: WorkspaceBench's `knowledge_cutoff` sampling and
the gate's, so O-lens reads of env transcripts are on text sampled like the gate's. Items are the
stable (≥ 7/8) items of bank v3 by default; `CUTOFF_QA_ALL=1` takes all 5,467, `CUTOFF_QA_BANK=v2` /
`v1` the older banks (every episode's meta records `bank`, and `category` for v3's entity-list
items). It labels the reply without an LLM judge: `correct` (normalized reply equals the truth or an
alias), `stale` (equals `qwen_modal`; for v1 items, the pre-cutoff analog), and `contains_truth` /
`contains_stale` for replies that add words. Normalization (`env.norm`): lowercase ASCII, quotes and
punctuation stripped (decimal points kept), leading article dropped, digit-group commas, currency
and unit words stripped from numbers ("$32.5 million" → `32.5`), whitespace collapsed.

## Bank v3 (default): v2 plus 100 entity-list agents

v3 = the 803 v2 items (unchanged, same gate fields) plus 4,664 new present-tense items from 100
Sonnet subagents run the same afternoon, each given brief v4 and a disjoint list of famous entities
rather than a domain (`source_batch` = `v4-entities`, ids `e001-*` … `e100-*`, `category` from the
list: countries, companies, sports, awards, products, institutions, media, science). The agents
wrote 4,862 questions; 4,664 survived the clean (dated, numeric, answer-in-question and duplicate
items dropped), all gated the same way as v2 (8 samples at T 1 + greedy, thinking off, no system
prompt): `runs/cutoff_gate/2026-10-29_v4-entities` (4,616 items: 1,456 stable-right, 457
stable-wrong, 2,696 unstable, 7 refusal) and `2026-10-29_v4-entities-e087b` (one agent's redone
file, 48 items: 12 / 5 / 31 / 0). For v4 items `truth_now` equals `truth`: each answer was verified
on a Wikipedia page the agent fetched that day, so there was no separate recheck. The bank header's
`how` field says the same.

| v3                                  | items                                    |
| ----------------------------------- | ---------------------------------------- |
| total                               | 5,467 (803 from v2, 4,664 entity-list)   |
| stable (≥ 7/8), right               | 1,547 (v2 79, entity-list 1,468)         |
| stable, wrong (Qwen's stale answer) | 649 (v2 187, entity-list 462)            |
| unstable                            | 3,258                                    |
| refusal or null                     | 13                                       |
| stable numeric items (flagged)      | 26 (all from v2; the v4 brief bans them) |

Yield by category for the 4,664 entity-list items (stable / all, with the stable-wrong count):

| category     | lists | stable / all  | stable-wrong |
| ------------ | ----- | ------------- | ------------ |
| companies    | 25    | 618 / 1,208   | 109          |
| countries    | 25    | 440 / 1,206   | 131          |
| sports       | 15    | 297 / 736     | 105          |
| products     | 8     | 172 / 343     | 25           |
| media        | 8     | 157 / 382     | 39           |
| institutions | 8     | 131 / 334     | 29           |
| awards       | 8     | 68 / 355      | 20           |
| science      | 3     | 47 / 100      | 4            |
| all          | 100   | 1,930 / 4,664 | 462          |

Per list, the worst yields were the award lists other than the Oscars: Booker / Pulitzer / Hugo /
Nebula / Newbery (`e070`, 0 of 50), Time Person of the Year / richest person / Pritzker / Goldman /
Kyoto (`e071`, 0 of 16 written), Miss Universe / Words of the Year / Game Awards (`e073`, 2 of 50),
the Grammys' big four / Brit / Mercury / Eurovision (`e068`, 3 of 50), Emmys / Golden Globes / Palme
d'Or / Tonys (`e069`, 3 of 50), Laureus / Heisman / MVPs (`e072`, 6 of 40); also newspaper editors
(`e092`, 7 of 48), cycling grand tours (`e062`, 7 of 50) and tennis slams (`e060`, 8 of 49). The
best were household-name companies (Apple / Microsoft / Alphabet `e026` 32 of 49, UnitedHealth / J&J
/ P&G `e027` 33 of 50, LVMH / Novo Nordisk / ASML `e028` 34 of 49, UMG / Live Nation / Nike `e038`
34 of 47), browsers and flagship products (`e078`, 34 of 45), US major-league sports (NFL `e056` 34
of 50, NBA `e057` 30 of 49, MLB `e058` 29 of 50) and the Oscars (`e067`, 32 of 50). Annual winners
are exactly what a January-2025 model can't hold stably; offices and flagships it can.

### What went wrong in the 100-agent run

- **Gate crash on `event_date: null`.** Brief v4 lets an unchanged item carry `event_date: null`;
  `scripts/cutoff_gate.py` sliced `q["event_date"][:4]` for the by-year table and crashed on the
  first one. Fixed in this PR (`year` is `None`, tallied as `null` in `by_year`): 2,969 of the 4,664
  items have no date.
- **The per-session WebSearch cap.** Claude Code gives a session 1,000 WebSearch calls shared by
  every subagent it spawns, refilling at 100 per hour (`CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION`).
  The first v4 agents searched for each fact and drained it within minutes, so later agents got
  search errors and stalled. The fix is in the brief: WebFetch the entity's Wikipedia page first
  (guess the URL), and at most 8 WebSearch calls per agent. 100 × 8 = 800 fits the budget.
- **Two empty lists.** The first cut of the country pool ran out before its last two chunks, so
  lists 24 and 25 were handed out empty (the agents wrote nothing). They were replaced with two
  subnational lists (US governors and big-city mayors; Canadian and Australian premiers, UK first
  ministers, German minister-presidents, Indian chief ministers), which yielded 25 of 45 and 24 of
  49 stable. `scripts/cutoff_facts/entity_lists.md` has the list build.
- **Short files.** Seven agents wrote fewer than 40 questions (`e071` 16, `e099` 10, `e089` 25,
  `e023` 30, `e088` 30, `e077` 35, `e079` 39): tiny countries and single-office bodies run out of
  famous things to ask. One agent's file (`e087`) was redone and gated separately (`_e087b`).

### What to do next time

- Assign entities, not domains: a domain brief sends a hundred agents to the same ten facts.
- Skip the low-yield award categories (anything whose answer is "the most recent winner of an annual
  prize"); keep the Oscars at most. Spend the agents on companies, flagship products, offices and US
  major-league sports.
- No dates in the question, no numeric answers (both already in the brief; the clean still dropped
  some).
- WebFetch first, 8 searches per agent at most, and watch the session's search budget.
- At most 50 concurrent subagents: 100 at once hit the search cap and the tool-call rate together.
- Lists of 7–16 entities are enough for 50 questions when each has several famous offices; give a
  list of tiny or single-office entities more members (Monaco / Andorra / San Marino got 30).

## Bank v2

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

- **Numeric truths** (v1 135, v2 60, 26 of them stable): see the rules above; `numeric` flags them
  (dates like "february 2026" and version-ish names like "a19 pro" are not flagged).
- **Stale analogs are partly from memory.** The v1 brief asked for the analog but not for its
  verification, so many are the subagent's recollection, and 109 are prose ("Bernard Arnault was
  also #1 on the 2023 list") rather than a value. In v2 the gate's `qwen_modal` replaces the analog
  as the stale label, which is the point.
- **Truth drifts.** A present-tense question's answer keeps changing; `truth` is the writer's value
  at authoring time and `truth_now` the 2026-10-29 recheck (13 of 266 stable v2 items had changed, 3
  could not be verified; v3's entity-list items were verified on a fetched page the same day, so
  `truth_now` = `truth` with no second look). Re-check before quoting accuracy.
- **Fetched pages were summarizer digests.** The subagents' WebFetch returned a model-written
  summary of the page, not its text, so a "fetched page" verification is one more model reading the
  source. The "two search results" items rest on snippets agreeing.
- 23 v1 `sci-*` questions end with a phrase that differs from their `answer_type` field ("scientific
  name" vs `species name`); harmless for the env, which sends the question as is.
- The pilot's 94 items (`b1-*`) are snippet-verified and use coarser domains than the batch.

## How to generate more

1. Write the avoid-list: every existing question with its answer (one line each) from the current
   bank, so new agents don't ask about the same thing or answer.
2. Write one entity list per agent (disjoint, 7–16 famous entities with a "what to ask" paragraph;
   [`scripts/cutoff_facts/entity_lists.md`](../../scripts/cutoff_facts/entity_lists.md)), skipping
   the categories above that failed.
3. One Sonnet subagent per list, at most 50 at a time, 50 questions each, given
   [`scripts/cutoff_facts/BRIEF.md`](../../scripts/cutoff_facts/BRIEF.md) (v4: present tense, no
   date, famous entities from the list, WebFetch first with at most 8 searches, no numbers, the
   writer's predicted stale value as a bare `pre_cutoff_analog`), the list and the avoid-list,
   writing one JSONL file per agent with the brief's exact keys.
4. `uv run python scripts/cutoff_facts/merge.py --batch DIR/*.jsonl` and read its output: fix
   invalid rows, decide on leaks and shared truths. (v3's own clean and gate-attach ran as scratch
   scripts that were not kept; `merge.py` still writes v1, so port its checks or redo them by hand.)
5. Manual same-event dedup from the Markdown table; add the dropped ids to `DROPPED`, bump the bank
   version in `merge.py` and `--upload`.
6. Gate the new items with `scripts/cutoff_gate.py` (8 samples at T 1 + greedy, thinking off, no
   system prompt) and attach `qwen_modal` / `gate_class`; recheck the stable items' truth on the web
   and record `truth_now`.
