# cutoff_qa: dated factual questions across Qwen's knowledge cutoff

**Status (2026-10-29):** ACTIVE. Bank v1 on HF and the `nohint` env are in; the Qwen stability gate
is running separately (numbers below are a placeholder); the `hint` arm is not built.

Env: [`environments/cutoff_qa/`](../../environments/cutoff_qa/). Bank:
[`banks/cutoff_qa/v1.json`](https://huggingface.co/datasets/Robbie-Kylie-NN-sprint/counterfactual-investigator-agent-with-olens/tree/main/banks/cutoff_qa)
on the sprint HF dataset, with the subagents' brief beside it (`BRIEF.md`; a copy is in
[`scripts/cutoff_facts/`](../../scripts/cutoff_facts/)). The env fetches the bank to
`data/banks/cutoff_qa/` on first use; it is never in git.

## What the bank is

1,072 short-answer questions about events from January 2024 to September 2026, for a subject
(Qwen3.6-27B) whose knowledge cutoff is about January 2025. A question pins one event by month and
year ("Which album won British Album of the Year at the Brit Awards in February 2026?") and ends "If
you are not sure, give your best guess. Answer with just the `<answer_type>`." Each item has a
lowercase canonical `truth`, `truth_aliases` (other surface forms), and for 627 items (58%) a
`pre_cutoff_analog`: the older value a Jan-2025 model is likely to give (the 2025 winner, the
previous CEO). Also `domain`, `event_date` (`YYYY-MM`), `answer_type`, `source_url`, `verified_by`,
`notes`, and `numeric` (the truth is digits only, `^[\d.,% ]+$`).

| split                                   | count                              |
| --------------------------------------- | ---------------------------------- |
| events in 2024 / 2025 / 2026            | 320 / 531 / 221                    |
| with a pre-cutoff analog                | 627                                |
| numeric truth                           | 135 (see below)                    |
| verified by two agreeing search results | 594 (61% of the 978 batch-2 items) |
| verified on a fetched page              | 384 (39%)                          |
| pilot items, snippet-verified           | 94 (`b1-*`)                        |

The env asks each question as the only user message, no system prompt, thinking off
(WorkspaceBench's knowledge*cutoff render), and labels the reply without an LLM judge: `correct`
(normalized reply equals the truth or an alias), `stale` (equals the analog), and
`contains*\*` for replies that add words. Normalization (`env.norm`): lowercase ASCII, quotes and punctuation stripped (decimal points kept), leading article dropped, digit-group commas, currency and unit words stripped from numbers ("$32.5 million" → `32.5`),
whitespace collapsed.

## How it was made (2026-10-29)

A 100-question pilot (5 Sonnet subagents: business, science, space, sports, tech), then 20 domains ×
50 questions, one Sonnet subagent each (AI, aviation and rail, auctions and records, automotive,
built infrastructure, corporate events, economics and markets, film and TV, football, video games,
health/internet/publishing, music, PC hardware, physical and life sciences, politics, annual
rankings, smartphones, software and standards, spaceflight, other sports). Every agent read
[`BRIEF.md`](../../scripts/cutoff_facts/BRIEF.md) and an avoid-list of every existing question,
started from Wikipedia "2024/2025/2026 in X" pages, list pages and release histories, and verified
each answer either on a page it fetched or by two independent search results agreeing (the pilot
accepted one result's snippet). The brief's rules: one event pinned by month and year, one short
unambiguous answer that a smart reader could not derive from the question (no year-tracking version
numbers, no "maker of Pringles"), one question per record / office / annual list, post-cutoff
questions preferring a stale trap with the old value recorded as the analog, nothing after
September 2026.

[`scripts/cutoff_facts/merge.py`](../../scripts/cutoff_facts/merge.py) merged the 25 JSONL files
(1,100 rows), checked the schema and dates, flagged questions containing their own truth (0), listed
truths shared by several items (37, e.g. Carlos Alcaraz three times: different events, kept) and
wrote the bank plus a Markdown table. Dedup was manual, by reading the table: 6 pilot rows dropped
in the pilot's own cleaning, then 22 same-event duplicates (the better-phrased one kept) and two
derivable answers dropped at the merge; the 28 ids are `DROPPED` in the script, so it reproduces the
bank byte for byte. Three pilot rows whose `domain` field held the question suffix (a subagent bug)
were repaired (`FIXES`).

## Known weaknesses

- **Numeric truths (135 items).** Robbie's decision (2026-10-29): no numeric answers in future
  datagen, for two reasons: Qwen's numbers vary across samples, so they fail the stability gate, and
  token lenses can't name multi-digit values because Qwen splits digits. The brief now says so; v1
  still contains them. Report them as a separate stratum or exclude them (`numeric` on each item;
  dates like "february 2026" and version-ish names like "a19 pro" are not flagged).
- **Stale analogs are partly from memory.** The brief asked for the analog but not for its
  verification, so many are the subagent's recollection, and 109 are prose ("Bernard Arnault was
  also #1 on the 2023 list") rather than a value; `stale` compares against the text before any
  parenthetical and never matches those. Treat `stale` as a lower bound on stale answers.
- **Fetched pages were summarizer digests.** The subagents' WebFetch returned a model-written
  summary of the page, not its text, so a "fetched page" verification is one more model reading the
  source. The "two search results" items rest on snippets agreeing.
- 23 `sci-*` questions end with a phrase that differs from their `answer_type` field ("scientific
  name" vs `species name`); harmless for the env, which sends the question string as is.
- The pilot's 94 items (`b1-*`) are snippet-verified and use coarser domains than the batch.

## Qwen stability gate

Being run separately with a standalone script against the shared Qwen server (k samples per
question, thinking off; which items Qwen answers consistently, correctly and with the stale value).
_Placeholder: numbers and the surviving-item list go here when that run finishes._

## How to generate more

1. Write the avoid-list: every existing question with its date and answer (one line each) from the
   current bank, so new agents don't ask about the same event or answer.
2. One Sonnet subagent per domain, 50 questions each, given
   [`BRIEF.md`](../../scripts/cutoff_facts/BRIEF.md) (now: no numeric answers) and the avoid-list,
   writing one JSONL file per agent with the brief's exact keys. New domains, or a second pass over
   an existing one with the old questions in the avoid-list.
3. `uv run python scripts/cutoff_facts/merge.py --batch DIR/*.jsonl` (plus `--pilot` for any
   snippet-verified batch) and read its output: fix invalid rows, decide on leaks and shared truths.
4. Manual same-event dedup from the Markdown table; add the dropped ids to `DROPPED`, bump the bank
   version in `merge.py` (`v2.json`, `HEADER["version"]`) and `--upload`.
5. Run the Qwen gate on the new items and record the survivors here.
