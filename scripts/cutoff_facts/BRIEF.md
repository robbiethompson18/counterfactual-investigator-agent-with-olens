# Brief v5: author 50 short-answer "knowledge cutoff" questions about your assigned entities

v5 (2026-10-29): the entity lists say what changed after January 2025; ask about those facts first
and stop when the list runs dry. v4: entity lists instead of domains, WebFetch first with a search
cap. v3: present tense, no dates. The v1 gate (1,072 dated questions) found Qwen3.6-27B holds a
stable belief on only 14% of them, because most were too obscure, and the same 174 famous questions
asked present-tense gave 32% stable-wrong (the stale trap firing) against 2% asked dated. So:
**famous things, asked as they stand today, with no date in the question.**

You are building a dataset of factual questions whose answer changed around January 2025, the
knowledge cutoff of the subject model (Qwen3.6-27B). Asked "Who is the CEO of Intel?" with no date,
a model with that cutoff confidently gives the old value; where nothing changed, it answers right.
Both are useful, changed facts more so. Your job: up to 50 questions, each verified as of today,
each with a short unambiguous answer.

## Per question

- **Present tense, no date, no event.** Ask about the current state of something: "Who is the
  president of X?", "Which club is the reigning Champions League winner?", "What is the newest
  iPhone?" Never "In March 2025, who was named ..."; never a month or year in the question. The
  question must read naturally today and would have read naturally in 2024.
- **Famous, or drop it.** The subject only gives a stable answer when both the current value and the
  pre-cutoff value are front-page, general-knowledge-quiz famous: a head of state, a household-name
  company's CEO or flagship product, a major championship, a top award's winner, a milestone that
  made world news. Test: would a well-read person in 2024 have known the OLD value without looking
  it up? If not, drop it. Niche examples that failed the gate: a metro line's terminus, a mid-size
  company's CEO, a regional cup winner, a Wikipedia tally.
- **Changed facts first.** Your entity list marks what changed about each entity after January 2025
  (a new CEO, a new champion, a new owner, a rename). Ask about those facts first, then other
  current facts about the same entities. Say in `notes` whether you believe the answer changed since
  January 2025. Don't pad: if the list runs dry at 30 good questions, stop at 30.
- **Record your predicted stale value.** For every question put in `pre_cutoff_analog` the single
  bare answer you expect a January-2025 model to give (no explanation, no parenthetical): for a
  changed item that is the old value, for an unchanged item it equals `truth`. It is the writer's
  prediction, not verified; the gate measures what Qwen actually says.
- **Your entities, not a domain.** You were given an explicit list of entities (companies,
  countries, competitions, awards, products, institutions). Write only about those. For each entity
  ask about its current state: who holds the office, who is the CEO or chair or coach, what the
  flagship or latest model is called, who the reigning champion or most recent winner is, where it
  is based. Several questions per entity is fine when each asks a different thing; spread across the
  list.
- **WebFetch first, search sparingly.** Fetch the entity's Wikipedia page directly (guess the URL:
  https://en.wikipedia.org/wiki/<Entity_Name>) and read the current holder from its infobox or lead;
  follow a redirect or the "List of ..." page when needed. Use WebSearch only when you cannot guess
  the page, and **at most 8 WebSearch calls in total**: the search budget is shared by a hundred
  agents and runs out. Load WebFetch / WebSearch via ToolSearch if they aren't loaded.
- **Verify it as of today.** The fetched page must state the current answer. Record the URL you read
  it on, and set `verified_by` to "fetched page". If you can't verify, drop it.
- **Short, unambiguous answer.** One name, one title, one place. No lists. Drop anything where two
  answers are defensible, where sources disagree, or where the value keeps changing (live prices,
  rankings that move weekly).
- **Not derivable from the question.** No "which company, maker of Pringles", no answers that follow
  from a generic rule. Ask yourself: could a smart reader who never heard of the thing guess it? If
  yes, drop it.
- **One question per series.** Don't ask the same office or competition several ways.
- **No numeric answers.** Every answer must be a name, title, place, product or word, never a
  number, year, price, count, version number or measurement. (Two reasons: the subject's numbers
  vary across samples so they fail the stability gate, and token lenses can't name multi-digit
  values because the subject splits digits.) If the natural question has a numeric answer, ask a
  different question about the same thing whose answer is a name, or drop it.
- **Don't repeat existing items.** The file `existing_list4.txt` beside this brief lists every
  question already in the dataset (question, answer). Don't ask about the same thing or the same
  answer.

## Output

Write ONE file at the path you were given, one JSON object per line, exactly these keys:

```
{"id": "<prefix>-01", "domain": "<your domain>", "event_date": "YYYY-MM of the most recent change, or null if unchanged since 2024",
 "question": "<the question, ending with: If you are not sure, give your best guess. Answer with just the <answer_type>.>",
 "answer_type": "<e.g. product name / company name / person name>",
 "truth": "<canonical short answer today, lowercase for names>",
 "truth_aliases": ["<other acceptable surface forms>"],
 "pre_cutoff_analog": "<the bare answer a Jan-2025 model would give>",
 "source_url": "<the url you read the answer on>",
 "verified_by": "fetched page" | "two search results",
 "notes": "<changed | unchanged, and one line on why this is verifiable and unambiguous>"}
```

**Append to the file after every 10 questions** (don't hold all 50 in memory until the end), so
partial work survives if you run out of room. Number ids sequentially.

Write nothing anywhere else: no files in the repo, no commits, no messages to anyone. When done,
reply with the count written, the changed/unchanged split, and a one-line list of questions you
dropped with the reason.
